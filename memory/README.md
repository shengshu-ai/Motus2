# Motus2 Memory: Long-Horizon Training with Variable-Length Packing

Memory and variable-length training infrastructure for the Motus2 video-action model. The policy
denoises actions chunk by chunk over a full episode while attending to a bounded memory of the
past:

- **anchor** latents: the first `memory.anchor_latents` latent frames stay visible for the whole
  episode;
- **recent** latents: the last `memory.recent_latents` latent frames are visible at full
  resolution;
- **gist** tokens: every latent frame is summarized into `memory.gist_tokens_per_frame` learnable
  tokens that remain visible after the frame leaves the recent window.

Setting `anchor_latents=0`, `gist_tokens_per_frame=0` and a very large `recent_latents` recovers
full-attention training over the episode.

Training packs whole trajectories of different lengths into one attention sequence (no padding):
`infra/varlen` builds the packed layout, block masks and RoPE, and the model runs FlexAttention
over the packed sequence. `data/length_bucketing.py` groups trajectories under a latent budget per
batch.

## Layout

```text
models/        Motus model with memory attention, gist tokens, RoPE, attention masks
infra/varlen/  packing, block masks and RoPE assembly for variable-length batches
train/         train.py (training entry), sample.py (in-training evaluation)
data/          RoboTwin and Tianji/Wuji loaders, length-bucketing sampler
utils/         action-space conventions, schedulers, checkpoint key conversion
third_party/wan/ vendored Wan2.2 modules (Apache-2.0)
configs/       example training configs and DeepSpeed configs
scripts/       launcher
```

## Setup

```bash
pip install -r requirements.txt
pip install "flash-attn>=2.8.3" --no-build-isolation   # optional, see requirements.txt
```

Download Wan2.2-TI2V-5B and point `model.wan.checkpoint_path` in your config to it. Training
consumes pre-computed Wan2.2 VAE latents and UMT5 text embeddings.

## Training

Example configs in `configs/`:

- `memory_rmbench_example.yaml`: MemoryWAM on RMBench (RoboTwin-format latents, joint actions)
  with 2 anchor latents, 4 recent latents and 8 gist tokens per frame, packed under a
  200-latent budget, initialized from a Stage 2 checkpoint.
- `memory_realrobot_example.yaml`: MemoryWAM on real-robot data (`tianji_wuji` loader,
  dual dexterous hands, `qpos20_rel6d` actions) with the same memory layout under a 100-latent
  budget. Full attention is `anchor_latents: 0`, `gist_tokens_per_frame: 0`,
  `recent_latents: 100000`.

Replace the `/path/to/...` placeholders, then launch from the `memory/` directory. `train/train.py`
takes only `--config`; the DeepSpeed config (`system.deepspeed_config`) and the run name
(`logging.run_name`) come from the YAML:

```bash
torchrun --nproc_per_node=8 train/train.py --config configs/memory_rmbench_example.yaml
```

Key options: `model.memory.{anchor_latents,gist_tokens_per_frame,recent_latents}`,
`model.action_mode` (`joint` or `qpos20_rel6d`), `model.action_rope_mode` (`video_shared` places
action registers on the video latent clock, `chunk_local` restarts action RoPE every chunk).
Checkpoints are released separately.

## Inference

`init_kv_cache`, `fill_kv_cache` / `fill_kv_cache_from_latent`, then `inference_step_streaming`
once per chunk; the cache keeps anchors, gist tokens and the recent window.

`inference/` contains a machine-independent HTTP policy server built on this interface and a
client example with a RoboTwin-style adapter; see `inference/README.md`:

```bash
python inference/server.py --config inference/config_example.yaml
```

## License

Apache License 2.0 (see `LICENSE`). Third-party code and its licenses are listed in `NOTICE`.
