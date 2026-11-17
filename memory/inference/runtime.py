"""Streaming model runtime behind the policy server (hybrid-memory model).

The runtime owns one Motus model and one episode at a time:

* the model is built from a release training config with
  ``train/train.py:create_model_and_optimizer`` and loaded with
  ``Motus.load_checkpoint_weights``;
* camera images are composed into the model's video input (see
  ``compose_frame``) and kept at video rate;
* observed frames are VAE-encoded (one initial frame, then 4 frames per
  latent) and appended to the KV cache with ``fill_kv_cache_from_latent``.
  The model itself decides what stays visible: the first
  ``memory.anchor_latents`` latents, every latent's gist tokens and the newest
  ``memory.recent_latents`` latents. The cache is never trimmed or rebuilt
  from the outside, because both the anchors and the absolute RoPE clock
  depend on the whole episode;
* ``infer`` runs ``Motus.inference_step_streaming`` on the current state and
  returns the action chunk with the wrist xyz normalization undone
  (``Motus._unscale_qpos20_rel_wrist_xyz``; a no-op for ``joint``).

Frame schedule per action chunk (K = fixed_chunk_size, 4 frames per latent):
the first /infer of an episode carries one frame that primes the cache. After
that, while a chunk of ``K * actions_per_latent`` actions executes, the client
sends ``4K - 1`` frames with /update_obs, one every ``actions_per_frame``
actions, and the frame taken after the last action rides with the next /infer.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "train", _REPO_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from train import create_model_and_optimizer, load_config  # noqa: E402

from protocol import Action, Observation  # noqa: E402

logger = logging.getLogger(__name__)

_INTERPOLATION = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _letterbox(image: np.ndarray, size_hw, interpolation: int) -> np.ndarray:
    """Resize keeping the aspect ratio, then center-pad with black to size_hw."""
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    src_h, src_w = image.shape[:2]
    scale = min(target_h / src_h, target_w / src_w)
    new_h, new_w = max(1, int(src_h * scale)), max(1, int(src_w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    out = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    y0, x0 = (target_h - new_h) // 2, (target_w - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def compose_frame(images: Dict[str, np.ndarray], vision_cfg: Dict[str, Any], size_hw) -> np.ndarray:
    """Compose named camera images into the model input, [3, H, W] float32 in [0, 1].

    The model is single-view; ``vision.layout`` must match how the training
    latents were encoded:

    * ``views``: exactly one camera (or a frame the client already composed),
      letterboxed to (video_height, video_width).
    * ``grid``: ``grid: [rows, cols]`` cameras, row-major in ``cameras``
      order, each resized to ``cell_size: [h, w]``, stitched, and the whole
      canvas letterboxed to the model size.
    * ``tshape``: the first camera on top at its own size, the remaining
      cameras side by side below it at half height.
    """
    cameras = list(vision_cfg["cameras"])
    missing = [name for name in cameras if name not in images]
    if missing:
        raise ValueError(f"missing camera images {missing}; got {sorted(images)}")
    frames = [np.asarray(images[name]) for name in cameras]
    for name, frame in zip(cameras, frames):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"camera {name!r} must be HxWx3 uint8, got {frame.shape} {frame.dtype}")
    layout = str(vision_cfg.get("layout", "views"))

    if layout == "views":
        if len(frames) != 1:
            raise ValueError(f"layout=views needs exactly 1 camera for this model, got {len(frames)}")
        interp = _INTERPOLATION[str(vision_cfg.get("interpolation") or "linear")]
        view = _letterbox(frames[0].astype(np.float32) / 255.0, size_hw, interp)
    elif layout == "grid":
        rows, cols = (int(v) for v in vision_cfg["grid"])
        cell_h, cell_w = (int(v) for v in vision_cfg["cell_size"])
        if len(frames) != rows * cols:
            raise ValueError(f"layout=grid needs {rows * cols} cameras, got {len(frames)}")
        interp = _INTERPOLATION[str(vision_cfg.get("interpolation") or "area")]
        cells = [cv2.resize(f, (cell_w, cell_h), interpolation=interp) for f in frames]
        canvas = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        view = _letterbox(canvas, size_hw, interp).astype(np.float32) / 255.0
    elif layout == "tshape":
        if len(frames) < 2:
            raise ValueError("layout=tshape needs at least 2 cameras")
        interp = _INTERPOLATION[str(vision_cfg.get("interpolation") or "linear")]
        top = frames[0]
        cell_w, cell_h = top.shape[1] // (len(frames) - 1), top.shape[0] // 2
        bottom = np.hstack([cv2.resize(f, (cell_w, cell_h)) for f in frames[1:]])
        canvas = np.vstack([top[:, :bottom.shape[1]], bottom])
        view = _letterbox(canvas.astype(np.float32) / 255.0, size_hw, interp)
    else:
        raise ValueError(f"unknown vision.layout {layout!r}")
    return np.ascontiguousarray(view.transpose(2, 0, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], device: str):
    """Build the release model from its training config and load the checkpoint."""
    model_cfg = cfg["model"]
    config = load_config(str(model_cfg["training_config"]))
    wan_path = Path(model_cfg["wan_path"]).expanduser()
    # The checkpoint carries every trained weight; only the WAN config and VAE
    # are read from wan_path.
    config.model.load_pretrained_backbones = False
    config.model.wan.checkpoint_path = str(wan_path)
    config.model.wan.config_path = str(wan_path)
    config.model.wan.vae_path = str(wan_path / "Wan2.2_VAE.pth")
    config.model.wan.gradient_checkpointing = False
    if model_cfg.get("action_stats_path"):
        config.dataset.action_stats_path = str(model_cfg["action_stats_path"])
    model, optimizer, scheduler = create_model_and_optimizer(config, torch.device(device))
    del optimizer, scheduler
    model.load_checkpoint_weights(str(model_cfg["checkpoint"]), strict=bool(model_cfg.get("strict_load", True)))
    model.eval()
    return model, config


class TextEncoder:
    """UMT5 encoder of Wan2.2, loaded on first use; embeddings cached per instruction."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self._encoder = None
        self._cache: Dict[str, torch.Tensor] = {}

    def __call__(self, instruction: str) -> torch.Tensor:
        if instruction not in self._cache:
            if self._encoder is None:
                from wan.modules.t5 import T5EncoderModel

                self._encoder = T5EncoderModel(
                    text_len=512,
                    dtype=torch.bfloat16,
                    device=self.device,
                    checkpoint_path=self.checkpoint_path,
                    tokenizer_path=self.tokenizer_path,
                )
            with torch.no_grad():
                out = self._encoder([instruction], self.device)
            emb = out[0] if isinstance(out, (list, tuple)) else out
            self._cache[instruction] = emb.squeeze(0) if emb.dim() == 3 else emb
        return self._cache[instruction]


def load_language_embedding(data: bytes) -> torch.Tensor:
    """Load a precomputed UMT5 embedding ([L, 4096] tensor, or a 1-element list)."""
    obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("embedding", "emb", "t5_embedding"):
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, (list, tuple)):
        if len(obj) != 1:
            raise ValueError(f"expected a single language embedding, got a list of {len(obj)}")
        obj = obj[0]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"unsupported language embedding payload {type(obj)!r}")
    if obj.dim() == 3 and obj.shape[0] == 1:
        obj = obj.squeeze(0)
    if obj.dim() != 2:
        raise ValueError(f"language embedding must be [L, D], got {tuple(obj.shape)}")
    return obj


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class MotusRuntime:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = str(cfg["model"].get("device", "cuda"))
        self.model, self.train_config = build_model(cfg, self.device)
        mc = self.model.config
        self.dtype = self.model.dtype
        self.size_hw = (int(mc.video_height), int(mc.video_width))
        self.vision_cfg = dict(cfg["vision"])
        self.chunk_latents = int(mc.fixed_chunk_size)
        self.frames_per_chunk = 4 * self.chunk_latents
        self.actions_per_frame = int(mc.actions_per_latent) // 4

        infer_cfg = dict(cfg.get("inference") or {})
        self.num_inference_steps = infer_cfg.get("num_inference_steps")

        lang_cfg = dict(cfg.get("language") or {})
        wan_path = Path(cfg["model"]["wan_path"]).expanduser()
        self.text_encoder = TextEncoder(
            str(lang_cfg.get("t5_checkpoint") or wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(lang_cfg.get("t5_tokenizer") or wan_path / "google" / "umt5-xxl"),
            str(lang_cfg.get("device") or self.device),
        )
        self.embeddings_dir = (
            Path(lang_cfg["embeddings_dir"]).expanduser().resolve()
            if lang_cfg.get("embeddings_dir") else None
        )
        self.lang_emb: Optional[List[torch.Tensor]] = None
        self.instruction: Optional[str] = None

        self._model_lock = threading.RLock()
        self.reset()
        if lang_cfg.get("default_instruction"):
            self.set_language(instruction=str(lang_cfg["default_instruction"]))

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> dict:
        mc = self.model.config
        return {
            "action_mode": str(mc.action_mode),
            "action_dim": int(mc.action_dim),
            "state_dim": int(mc.action_state_dim),
            "chunk_length": self.chunk_latents * int(mc.actions_per_latent),
            "frames_per_chunk": self.frames_per_chunk,
            "cache_updates_per_chunk": self.frames_per_chunk - 1,
            "actions_per_frame": self.actions_per_frame,
            "cameras": list(self.vision_cfg["cameras"]),
            "layout": str(self.vision_cfg.get("layout", "views")),
            "num_views": 1,
            "video_size": list(self.size_hw),
            "memory": {
                "anchor_latents": int(mc.memory_anchor_latents),
                "gist_tokens_per_frame": int(mc.memory_gist_tokens_per_frame),
                "recent_latents": int(mc.memory_recent_latents),
            },
        }

    def health(self) -> dict:
        return {
            "language_set": self.lang_emb is not None,
            "instruction": self.instruction,
            "step": self._step,
            "primed": not self._is_first_step,
            "pending_frames": len(self._frame_buffer),
            "latents_seen": self._latents_seen,
        }

    # -- episode state -----------------------------------------------------

    def reset(self, session_epoch: Optional[str] = None) -> None:
        with self._model_lock:
            self.model.clear_kv_cache()
        self._frame_buffer: List[torch.Tensor] = []
        self._latents_seen = 0
        self._is_first_step = True
        self._current_frame: Optional[torch.Tensor] = None
        self._current_state: Optional[torch.Tensor] = None
        self._prepared_id: Optional[str] = None
        self._pending_ids: List[str] = []
        self._seen_ids: set = set()
        self._session_epoch = None if session_epoch is None else str(session_epoch)
        self._step = 0

    def set_language(self, instruction: Optional[str] = None,
                     embedding_bytes: Optional[bytes] = None,
                     embedding_path: Optional[str] = None) -> dict:
        """Select the episode's language embedding and reset the episode.

        Priority: uploaded embedding > embedding file under
        ``language.embeddings_dir`` > UMT5 encoding of ``instruction``.
        """
        if embedding_bytes is not None:
            emb, source = load_language_embedding(embedding_bytes), "upload"
        elif embedding_path:
            if self.embeddings_dir is None:
                raise ValueError("language.embeddings_dir is not configured")
            path = (self.embeddings_dir / embedding_path).resolve()
            if self.embeddings_dir not in path.parents:
                raise ValueError("embedding path must stay inside language.embeddings_dir")
            emb, source = load_language_embedding(path.read_bytes()), str(embedding_path)
        elif instruction:
            emb, source = self.text_encoder(str(instruction)), "umt5"
        else:
            raise ValueError("set_language needs an instruction or an embedding")
        self.lang_emb = [emb.to(device=self.device, dtype=self.dtype)]
        self.instruction = instruction
        self.reset()
        return {"source": source, "tokens": int(emb.shape[0])}

    # -- observations ------------------------------------------------------

    def accept(self, obs: Observation, *, cache_only: bool) -> str:
        """Check the observation against the episode's frame schedule."""
        observation_id = str(obs.meta.get("observation_id") or "")
        if not observation_id:
            raise ValueError("meta.observation_id is required")
        epoch = obs.meta.get("session_epoch")
        if self._session_epoch is not None and str(epoch) != self._session_epoch:
            raise ValueError("session_epoch does not match the last /reset; reset the episode")
        if observation_id in self._seen_ids:
            raise ValueError(f"observation_id reuse rejected: {observation_id}")
        if self.lang_emb is None:
            raise RuntimeError("language is not set; call /set_language first")
        updates = self.frames_per_chunk - 1
        if cache_only:
            if self._is_first_step:
                raise ValueError("/update_obs before the first /infer of the episode")
            if len(self._pending_ids) >= updates:
                raise ValueError(f"expected exactly {updates} /update_obs frames per chunk")
            self._pending_ids.append(observation_id)
        elif not self._is_first_step and len(self._pending_ids) != updates:
            raise ValueError(
                f"next /infer needs {updates} /update_obs frames, got {len(self._pending_ids)}"
            )
        self._seen_ids.add(observation_id)
        return observation_id

    def _frame_tensor(self, obs: Observation) -> torch.Tensor:
        frame = compose_frame(obs.images, self.vision_cfg, self.size_hw)
        return torch.from_numpy(frame).to(device=self.device, dtype=self.dtype)  # [3, H, W]

    def _state_tensor(self, state) -> torch.Tensor:
        if state is None:
            raise ValueError("state is required for inference")
        dim = int(self.model.config.action_state_dim)
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(f"state must have {dim} values, got {arr.shape[0]}")
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype).view(1, dim)

    def update_obs(self, obs: Observation) -> None:
        """Cache-only frame taken while the previous chunk executes."""
        self._frame_buffer.append(self._frame_tensor(obs))
        self._drain_frame_buffer()

    def prepare(self, obs: Observation) -> None:
        """Take the frame that conditions the next prediction (state comes later)."""
        frame = self._frame_tensor(obs)
        self._current_frame = frame
        if self._is_first_step:
            with self._model_lock:
                self.model.init_kv_cache(batch_size=1, start_frame_abs=0)
            self._append_latent(self._encode([frame], is_initial=True))
            self._is_first_step = False
        else:
            self._frame_buffer.append(frame)
            self._drain_frame_buffer()
        self._prepared_id = str(obs.meta["observation_id"])

    def set_state(self, state, prepared_id: str) -> None:
        if self._prepared_id is None:
            raise RuntimeError("/prepare_infer must come before /predict")
        if str(prepared_id) != self._prepared_id:
            raise ValueError(f"prepared observation mismatch: {prepared_id!r} != {self._prepared_id!r}")
        self._current_state = self._state_tensor(state)

    # -- KV cache ----------------------------------------------------------

    @torch.no_grad()
    def _encode(self, frames: List[torch.Tensor], is_initial: bool) -> torch.Tensor:
        """VAE-encode [3, H, W] frames in [0, 1] the way ``Motus.fill_kv_cache`` does.

        The initial frame goes through ``encode_chunk_first`` and stays a single
        latent (the memory model conditions on one real first latent); later
        calls take exactly 4 frames and reuse the VAE's causal feature cache.
        """
        video = torch.stack(frames, dim=1).unsqueeze(0)  # [1, 3, T, H, W]
        vae = self.model.video_model.vae
        x = (video * 2.0 - 1.0).to(self.dtype)
        with self._model_lock:
            if is_initial:
                vae.clear_cache()
                return vae.encode_chunk_first(x).to(self.dtype)
            return vae.encode_chunk(x).to(self.dtype)

    def _append_latent(self, latent: torch.Tensor) -> None:
        with self._model_lock:
            self.model.fill_kv_cache_from_latent(latent, language_embeddings=self.lang_emb)
        self._latents_seen += int(latent.shape[2])

    def _drain_frame_buffer(self) -> None:
        while len(self._frame_buffer) >= 4:
            latent = self._encode(self._frame_buffer[:4], is_initial=False)
            del self._frame_buffer[:4]
            self._append_latent(latent)

    # -- prediction --------------------------------------------------------

    @torch.no_grad()
    def predict(self) -> Action:
        if self._current_frame is None or self._current_state is None:
            raise RuntimeError("no prepared observation and state")
        with self._model_lock:
            video_latent, actions = self.model.inference_step_streaming(
                state=self._current_state,
                language_embeddings=self.lang_emb,
                num_inference_steps=self.num_inference_steps,
            )
            actions = self.model._unscale_qpos20_rel_wrist_xyz(actions.squeeze(0).float())
        del video_latent
        self._step += 1
        chunk = actions.cpu().numpy()
        self._pending_ids.clear()
        self._prepared_id = None
        return Action(actions=chunk, meta={"step": self._step, "chunk_length": int(chunk.shape[0])})

    def infer(self, obs: Observation) -> Action:
        self.prepare(obs)
        self.set_state(obs.state, self._prepared_id)
        return self.predict()


def load_server_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
