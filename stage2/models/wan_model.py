# WAN Video Diffusion Model wrapper.
# Unified VAE interface: training and inference use the same encode() method.
# VAE encode internally handles chunk-by-chunk encoding with feat_cache.

import os
import sys
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import WAN backbone from third_party/
WAN_ROOT = str((Path(__file__).parent.parent / "third_party").resolve())
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

from wan.modules.model import WanModel, sinusoidal_embedding_1d

from .vae import VAEWrapper, load_vae

from utils.checkpoint_converter import convert_diffusers_keys, strip_dit_prefix

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

logger = logging.getLogger(__name__)


class WanVideoModel(nn.Module):
    """WAN video model with unified VAE (training and inference use same encode method)."""

    def __init__(self, model_config: Dict[str, Any], vae_path: str,
                 device: str = "cuda", precision: str = "bfloat16"):
        super().__init__()
        self.device = torch.device(device)
        self.precision = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]

        # WAN transformer backbone
        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)

        # Unified VAE (training and inference use the same encode method)
        # Supports both .pth and diffusers formats with streaming cache
        logger.info(f"Loading VAE from: {vae_path}")
        self.vae = load_vae(vae_path, torch_dtype=self.precision, torch_device=self.device)

        logger.info(f"WanVideoModel: {sum(p.numel() for p in self.wan_model.parameters()):,} params")

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """Encode video pixels to latents. [B,C,T,H,W] -> [B,C',T',H',W']

        Unified method for training and inference.
        VAE encode internally handles chunk-by-chunk encoding with feat_cache.
        """
        return self.vae.encode(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixels. [B,C,T,H,W] -> [B,C',T',H',W']"""
        return torch.stack([self.vae.decode([lat])[0] for lat in video_latents])

    # --- Factory methods ---

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, vae_path: str = None,
                        config_path: str = None, device: str = "cuda", precision: str = "bfloat16",
                        gradient_checkpointing: bool = False) -> "WanVideoModel":
        """Load from pretrained WAN checkpoint.

        Automatically detects format and infers paths from checkpoint_path:
        - Diffusers format: directory with model_index.json
          - config_path = checkpoint_path/transformer
          - vae_path = checkpoint_path/vae
        - Standard format: directory with config.json
          - config_path = checkpoint_path
          - vae_path = checkpoint_path/Wan2.2_VAE.pth (if not specified)

        Supports multiple weight formats:
        1. Sharded safetensors: diffusion_pytorch_model.safetensors.index.json
        2. Single safetensors file: diffusion_pytorch_model.safetensors
        3. Single .bin file: diffusion_pytorch_model.bin
        """
        # Check if this is a diffusers format (has model_index.json)
        model_index_file = os.path.join(checkpoint_path, "model_index.json")
        transformer_path = checkpoint_path
        is_diffusers = os.path.exists(model_index_file)

        if is_diffusers:
            # Diffusers format: load from transformer/ subdirectory
            logger.info(f"Detected diffusers format, loading from transformer/ subdirectory")
            transformer_path = os.path.join(checkpoint_path, "transformer")
            if not os.path.exists(transformer_path):
                raise FileNotFoundError(f"Diffusers format detected but transformer/ directory not found at {transformer_path}")
            config_path = config_path or transformer_path
            # Auto-infer vae_path for diffusers format
            if vae_path is None:
                vae_path = os.path.join(checkpoint_path, "vae")
                logger.info(f"Auto-inferred VAE path: {vae_path}")
        else:
            # Standard format
            config_path = config_path or checkpoint_path
            # Auto-infer vae_path for standard format
            if vae_path is None:
                vae_path = os.path.join(checkpoint_path, "Wan2.2_VAE.pth")
                if not os.path.exists(vae_path):
                    # Try alternative names
                    alt_paths = [
                        os.path.join(checkpoint_path, "vae.pth"),
                        os.path.join(checkpoint_path, "VAE.pth"),
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            vae_path = alt_path
                            logger.info(f"Auto-inferred VAE path: {vae_path}")
                            break
                    else:
                        logger.warning(f"VAE path not found, will try to load from: {vae_path}")
                else:
                    logger.info(f"Auto-inferred VAE path: {vae_path}")

        # Load config
        cfg_file = os.path.join(config_path, "config.json")
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"config.json not found at {cfg_file}")
        with open(cfg_file) as f:
            full_cfg = json.load(f)

        # Handle both diffusers format (in_channels/out_channels) and standard format (in_dim/out_dim)
        in_dim = full_cfg.get("in_dim") or full_cfg.get("in_channels", 16)
        out_dim = full_cfg.get("out_dim") or full_cfg.get("out_channels", 16)
        # Handle num_heads vs num_attention_heads
        num_heads = full_cfg.get("num_heads") or full_cfg.get("num_attention_heads", 24)
        # Handle dim (may not exist in diffusers format, need to compute from other params)
        dim = full_cfg.get("dim")
        if dim is None:
            # Diffusers format: dim = num_heads * attention_head_dim
            attention_head_dim = full_cfg.get("attention_head_dim", 128)
            dim = num_heads * attention_head_dim

        model_config = {
            "model_type": full_cfg.get("model_type", "i2v"),
            "patch_size": tuple(full_cfg.get("patch_size", (1, 2, 2))),
            "text_len": full_cfg.get("text_len", 512),
            "in_dim": in_dim,
            "dim": dim,
            "ffn_dim": full_cfg.get("ffn_dim", 12288),
            "freq_dim": full_cfg.get("freq_dim", 256),
            "text_dim": full_cfg.get("text_dim", 4096),
            "out_dim": out_dim,
            "num_heads": num_heads,
            "num_layers": full_cfg.get("num_layers", 30),
            "window_size": tuple(full_cfg.get("window_size", (-1, -1))),
            "qk_norm": full_cfg.get("qk_norm", True),
            "cross_attn_norm": full_cfg.get("cross_attn_norm", True),
            "eps": full_cfg.get("eps", 1e-6),
            "gradient_checkpointing": full_cfg.get("gradient_checkpointing", gradient_checkpointing),
        }

        model = cls(model_config, vae_path, device=device, precision=precision)

        # Load WAN weights from transformer_path (diffusers format) or checkpoint_path (standard format)
        weight_path = transformer_path if os.path.exists(model_index_file) else checkpoint_path

        # Check for sharded safetensors first (diffusion_pytorch_model.safetensors.index.json)
        index_file = os.path.join(weight_path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_file) and safe_load_file is not None:
            # Load sharded safetensors
            logger.info(f"Loading sharded safetensors from {weight_path}")
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # Collect all unique shard files
            shard_files = set(weight_map.values())
            sd = {}
            for shard_file in sorted(shard_files):  # Sort for consistent loading order
                shard_path = os.path.join(weight_path, shard_file)
                if os.path.exists(shard_path):
                    logger.debug(f"Loading shard: {shard_file}")
                    shard_sd = safe_load_file(shard_path)
                    sd.update(shard_sd)
                else:
                    logger.warning(f"Shard file not found: {shard_path}")
            if not sd:
                logger.warning(f"No weights loaded from sharded safetensors at {weight_path}")
                return model
        else:
            # Try single safetensors file
            ckpt_file = os.path.join(weight_path, "diffusion_pytorch_model.safetensors")
            if os.path.exists(ckpt_file) and safe_load_file is not None:
                logger.info(f"Loading single safetensors file from {ckpt_file}")
                sd = safe_load_file(ckpt_file)
            else:
                # Try .bin file
                bin_file = os.path.join(weight_path, "diffusion_pytorch_model.bin")
                if os.path.exists(bin_file):
                    logger.info(f"Loading .bin file from {bin_file}")
                    sd = torch.load(bin_file, map_location="cpu")
                else:
                    logger.warning(f"No checkpoint found at {weight_path}")
                    return model

        sd = strip_dit_prefix(sd)
        # Convert diffusers format keys if needed
        if is_diffusers:
            sd = convert_diffusers_keys(sd)
        missing, unexpected = model.wan_model.load_state_dict(sd, strict=False)
        if missing:
            logger.info(f"Missing keys: {len(missing)} keys (showing first 5: {missing[:5]})")
        if unexpected:
            logger.info(f"Unexpected keys: {len(unexpected)} keys (showing first 5: {unexpected[:5]})")
        logger.info("WAN weights loaded successfully")
        return model

    @classmethod
    def from_config(cls, config_path: str, vae_path: str,
                    device: str = "cuda", precision: str = "bfloat16") -> "WanVideoModel":
        """Init from config only (no pretrained weights)."""
        cfg_file = os.path.join(config_path, "config.json")
        with open(cfg_file) as f:
            full_cfg = json.load(f)

        # Handle both diffusers format (in_channels/out_channels) and standard format (in_dim/out_dim)
        in_dim = full_cfg.get("in_dim") or full_cfg.get("in_channels", 16)
        out_dim = full_cfg.get("out_dim") or full_cfg.get("out_channels", 16)
        # Handle num_heads vs num_attention_heads
        num_heads = full_cfg.get("num_heads") or full_cfg.get("num_attention_heads", 24)
        # Handle dim (may not exist in diffusers format, need to compute from other params)
        dim = full_cfg.get("dim")
        if dim is None:
            # Diffusers format: dim = num_heads * attention_head_dim
            attention_head_dim = full_cfg.get("attention_head_dim", 128)
            dim = num_heads * attention_head_dim

        model_config = {
            "model_type": full_cfg.get("model_type", "i2v"),
            "patch_size": tuple(full_cfg.get("patch_size", (1, 2, 2))),
            "text_len": full_cfg.get("text_len", 512),
            "in_dim": in_dim,
            "dim": dim,
            "ffn_dim": full_cfg.get("ffn_dim", 12288),
            "freq_dim": full_cfg.get("freq_dim", 256),
            "text_dim": full_cfg.get("text_dim", 4096),
            "out_dim": out_dim,
            "num_heads": num_heads,
            "num_layers": full_cfg.get("num_layers", 30),
            "window_size": tuple(full_cfg.get("window_size", (-1, -1))),
            "qk_norm": full_cfg.get("qk_norm", True),
            "cross_attn_norm": full_cfg.get("cross_attn_norm", True),
            "eps": full_cfg.get("eps", 1e-6),
            "gradient_checkpointing": full_cfg.get("gradient_checkpointing", False),
        }
        return cls(model_config, vae_path, device=device, precision=precision)
