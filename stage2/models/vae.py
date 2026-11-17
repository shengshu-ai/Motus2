# VAE wrapper supporting both .pth and diffusers formats with streaming cache.

import os
import torch
import torch.nn as nn
from typing import List, Optional
from diffusers import AutoencoderKLWan


def patchify(x, patch_size):
    """Rearrange spatial dims into channel dim for VAE input."""
    if patch_size is None or patch_size == 1:
        return x
    B, C, F, H, W = x.shape
    x = x.view(B, C, F, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(B, C * patch_size * patch_size, F, H // patch_size, W // patch_size)
    return x


class VAEWrapper:
    """VAE wrapper supporting both .pth and diffusers formats with streaming cache.

    Training and inference use the same encode() method.
    VAE encode internally handles chunk-by-chunk encoding with feat_cache.

    Interface:
    - encode(videos): [B, C, T, H, W] -> [B, C', T', H', W']
    - decode(zs): List[[C, T, H, W]] -> List[[C, T, H, W]]
    """

    def __init__(self, vae_model, vae_type: str = "diffusers"):
        """
        Args:
            vae_model: Either AutoencoderKLWan (diffusers) or VideoVAE_ (.pth format)
            vae_type: "diffusers" or "pth"
        """
        self.vae = vae_model
        self.vae_type = vae_type
        _params = vae_model.model.parameters() if vae_type == "pth" else vae_model.parameters()
        p = next(_params)
        self.device = p.device
        self.dtype = p.dtype

        if vae_type == "diffusers":
            # Diffusers format: use encoder and quant_conv directly
            self.encoder = vae_model.encoder
            self.quant_conv = vae_model.quant_conv

            # Load latents normalization parameters
            if hasattr(vae_model.config, "latents_mean") and hasattr(vae_model.config, "latents_std"):
                self.latents_mean = torch.tensor(vae_model.config.latents_mean, device=self.device, dtype=self.dtype)
                self.latents_std = torch.tensor(vae_model.config.latents_std, device=self.device, dtype=self.dtype)
            else:
                self.latents_mean = None
                self.latents_std = None

            # Count CausalConv3d layers for cache initialization
            if hasattr(self.vae, "_cached_conv_counts"):
                self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
            else:
                count = 0
                for m in self.encoder.modules():
                    if m.__class__.__name__ == "WanCausalConv3d":
                        count += 1
                self.enc_conv_num = count
        else:
            # .pth format: use encoder and conv1 directly for streaming support
            # Wan2_2_VAE's encode() clears cache each time (1+T/4), so we implement streaming ourselves
            self.model = vae_model.model  # WanVAE_ instance
            self.encoder = self.model.encoder  # Encoder3d
            self.conv1 = self.model.conv1  # CausalConv3d(z_dim * 2, z_dim * 2, 1)
            self.scale = vae_model.scale  # [mean, 1.0 / std]
            self.z_dim = self.model.z_dim  # Wan2_2_VAE doesn't store z_dim, get from model

            # Count CausalConv3d layers for cache
            if hasattr(self.model, "_enc_conv_num"):
                self.enc_conv_num = self.model._enc_conv_num
            else:
                # Count manually
                count = 0
                for m in self.encoder.modules():
                    if m.__class__.__name__ == "CausalConv3d":
                        count += 1
                self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        """Reset streaming state. Call between independent sequences."""
        self.feat_cache = [None] * self.enc_conv_num

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos to latents. [B, C, T, H, W] -> [B, C', T', H', W']

        VAE encode internally handles chunk-by-chunk encoding with feat_cache.
        Works for both training and inference.
        """
        self.clear_cache()  # Must reset between batches; otherwise cache from prev batch causes temporal=2 at first chunk
        with torch.no_grad():
            if self.vae_type == "diffusers":
                # Diffusers format: chunk-by-chunk encoding with feat_cache
                B, C, T, H, W = videos.shape
                iter_ = 1 + (T - 1) // 4

                # Apply patchify if needed (before chunking, same as pth format)
                patch_size = None
                if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
                    patch_size = self.vae.config.patch_size

                if patch_size is not None:
                    videos = patchify(videos, patch_size)

                # First chunk: [B, C, 1, H, W]
                x_first = videos[:, :, :1, :, :]
                feat_idx = [0]
                out = self.encoder(x_first, feat_cache=self.feat_cache, feat_idx=feat_idx)
                out = self.quant_conv(out)
                # Diffusers format: chunk into mu and log_var, normalize mu
                mu, log_var = out.chunk(2, dim=1)
                # Normalize mu using latents_mean and latents_std
                if self.latents_mean is not None and self.latents_std is not None:
                    mu = (mu - self.latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / self.latents_std.view(1, -1, 1, 1, 1))
                latents_list = [mu]

                # Process remaining chunks: 4 frames per chunk
                for i in range(1, iter_):
                    x_chunk = videos[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :]
                    feat_idx = [0]
                    out_chunk = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
                    out_chunk = self.quant_conv(out_chunk)
                    # Diffusers format: chunk into mu and log_var, normalize mu
                    mu, log_var = out_chunk.chunk(2, dim=1)
                    # Normalize mu using latents_mean and latents_std
                    if self.latents_mean is not None and self.latents_std is not None:
                        mu = (mu - self.latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / self.latents_std.view(1, -1, 1, 1, 1))
                    latents_list.append(mu)

                latents = torch.cat(latents_list, dim=2)
                return latents
            else:
                # .pth format: chunk-by-chunk encoding with feat_cache (same as diffusers format)
                # Wan2_2_VAE has float32 weights; use autocast (output float32)
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    B, C, T, H, W = videos.shape
                    iter_ = 1 + (T - 1) // 4
                    # Apply patchify (same as Wan2_2_VAE.encode())
                    videos = patchify(videos, patch_size=2)
                    # First chunk: [B, C, 1, H, W]
                    x_first = videos[:, :, :1, :, :]
                    feat_idx = [0]
                    out = self.encoder(x_first, feat_cache=self.feat_cache, feat_idx=feat_idx)
                    out = self.conv1(out)
                    # Chunk into mu and log_var, normalize mu (same as Wan2_2_VAE.encode())
                    mu, log_var = out.chunk(2, dim=1)
                    # Apply scale normalization (same as Wan2_2_VAE.encode())
                    if isinstance(self.scale[0], torch.Tensor):
                        mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(1, self.z_dim, 1, 1, 1)
                    else:
                        mu = (mu - self.scale[0]) * self.scale[1]
                    latents_list = [mu]
                    # Process remaining chunks: 4 frames per chunk
                    for i in range(1, iter_):
                        x_chunk = videos[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :]
                        feat_idx = [0]
                        out_chunk = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
                        out_chunk = self.conv1(out_chunk)
                        # Chunk into mu and log_var, normalize mu
                        mu, log_var = out_chunk.chunk(2, dim=1)
                        # Apply scale normalization
                        if isinstance(self.scale[0], torch.Tensor):
                            mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(1, self.z_dim, 1, 1, 1)
                        else:
                            mu = (mu - self.scale[0]) * self.scale[1]
                        latents_list.append(mu)
                    latents = torch.cat(latents_list, dim=2)
                    return latents

    def encode_chunk_first(self, video_frame: torch.Tensor) -> torch.Tensor:
        """Encode first frame (1 frame -> 1 latent) for streaming inference.

        Caller must call clear_cache() before the first use in an episode.

        Args:
            video_frame: [B, C, 1, H, W] - exactly 1 frame

        Returns:
            latent: [B, C', 1, H', W'] - 1 latent
        """
        with torch.no_grad():
            B, C, T, H, W = video_frame.shape
            assert T == 1, f"encode_chunk_first expects 1 frame, got {T}"
            if self.vae_type == "diffusers":
                patch_size = None
                if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
                    patch_size = self.vae.config.patch_size
                if patch_size is not None:
                    video_frame = patchify(video_frame, patch_size)
                feat_idx = [0]
                out = self.encoder(video_frame, feat_cache=self.feat_cache, feat_idx=feat_idx)
                out = self.quant_conv(out)
                mu, log_var = out.chunk(2, dim=1)
                if self.latents_mean is not None and self.latents_std is not None:
                    mu = (mu - self.latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / self.latents_std.view(1, -1, 1, 1, 1))
                return mu
            else:
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    video_frame = patchify(video_frame, patch_size=2)
                    feat_idx = [0]
                    out = self.encoder(video_frame, feat_cache=self.feat_cache, feat_idx=feat_idx)
                    out = self.conv1(out)
                    mu, log_var = out.chunk(2, dim=1)
                    if isinstance(self.scale[0], torch.Tensor):
                        mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(1, self.z_dim, 1, 1, 1)
                    else:
                        mu = (mu - self.scale[0]) * self.scale[1]
                    return mu

    def encode_chunk(self, video_chunk: torch.Tensor) -> torch.Tensor:
        """Encode a chunk of 4 frames to 1 latent (for streaming inference).

        Args:
            video_chunk: [B, C, 4, H, W] - exactly 4 frames

        Returns:
            latent: [B, C', 1, H', W'] - 1 latent

        Note: This method mirrors the logic in encode() for processing 4-frame chunks,
        but is optimized for streaming inference (single chunk, reuses feat_cache).
        """
        with torch.no_grad():
            B, C, T, H, W = video_chunk.shape
            assert T == 4, f"encode_chunk expects exactly 4 frames, got {T}"

            if self.vae_type == "diffusers":
                # Diffusers format: same logic as encode() for 4-frame chunks
                # Apply patchify if needed (before encoding)
                patch_size = None
                if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
                    patch_size = self.vae.config.patch_size

                if patch_size is not None:
                    video_chunk = patchify(video_chunk, patch_size)

                # Encode 4 frames -> 1 latent (using feat_cache for streaming)
                feat_idx = [0]
                out = self.encoder(video_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
                out = self.quant_conv(out)
                # Diffusers format: chunk into mu and log_var, normalize mu
                mu, log_var = out.chunk(2, dim=1)
                # Normalize mu using latents_mean and latents_std
                if self.latents_mean is not None and self.latents_std is not None:
                    mu = (mu - self.latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / self.latents_std.view(1, -1, 1, 1, 1))
                return mu
            else:
                # .pth format: same logic as encode() for 4-frame chunks
                # Wan2_2_VAE has float32 weights; use autocast (output float32)
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    video_chunk = patchify(video_chunk, patch_size=2)
                    # Encode 4 frames -> 1 latent (using feat_cache for streaming)
                    feat_idx = [0]
                    out = self.encoder(video_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
                    out = self.conv1(out)
                    # Chunk into mu and log_var, normalize mu (same as Wan2_2_VAE.encode())
                    mu, log_var = out.chunk(2, dim=1)
                    # Apply scale normalization (same as Wan2_2_VAE.encode())
                    if isinstance(self.scale[0], torch.Tensor):
                        mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(1, self.z_dim, 1, 1, 1)
                    else:
                        mu = (mu - self.scale[0]) * self.scale[1]
                    return mu

    def decode(self, zs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode latents to pixels. Accepts list of [C, T, H, W] tensors, returns list."""
        if not isinstance(zs, list):
            raise TypeError("zs should be a list")
        results = []
        with torch.no_grad():
            for z in zs:
                if self.vae_type == "diffusers":
                    # Add batch dimension: [C, T, H, W] -> [1, C, T, H, W]
                    z_batch = z.unsqueeze(0)
                    # Diffusers VAE decode returns a BaseDecoderOutput with .sample
                    decoded = self.vae.decode(z_batch)
                    if hasattr(decoded, 'sample'):
                        decoded = decoded.sample
                    # Remove batch dimension and clamp: [1, C, T, H, W] -> [C, T, H, W]
                    decoded = decoded.squeeze(0).float().clamp_(-1, 1)
                else:
                    # .pth format: use autocast(Wan2_2_VAE weights float32)
                    with torch.amp.autocast("cuda", dtype=torch.float32):
                        z_batch = z.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
                        decoded = self.model.decode(z_batch, self.scale)
                        decoded = decoded.squeeze(0).float().clamp_(-1, 1)  # [1, C, T, H, W] -> [C, T, H, W]
                results.append(decoded)
        return results


def load_vae(vae_path: str, torch_dtype=torch.bfloat16, torch_device="cuda"):
    """Load VAE from either .pth or diffusers format.

    Args:
        vae_path: Path to VAE checkpoint
            - If directory with config.json: diffusers format
            - If .pth file: VideoVAE_ format

    Returns:
        VAEWrapper instance
    """
    # Check if it's a diffusers format directory
    if os.path.isdir(vae_path) and os.path.exists(os.path.join(vae_path, "config.json")):
        # Diffusers format
        vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch_dtype)
        vae = vae.to(torch_device)
        return VAEWrapper(vae, vae_type="diffusers")
    elif vae_path.endswith(".pth"):
        # .pth format: use Wan2_2_VAE from bak (Wan2.2_VAE.pth, supports streaming)
        import sys
        from pathlib import Path
        bak_root = Path(__file__).parent.parent / "third_party"
        if str(bak_root) not in sys.path:
            sys.path.insert(0, str(bak_root))
        from wan.modules.vae2_2 import Wan2_2_VAE
        vae = Wan2_2_VAE(vae_pth=vae_path, device=torch_device)
        return VAEWrapper(vae, vae_type="pth")
    else:
        raise ValueError(f"Unsupported VAE format: {vae_path}. Expected .pth file or diffusers directory with config.json")
