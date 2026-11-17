# Motus2 Stage 1: Video Pretraining

Stage 1 of Motus2 continues training the Wan2.2-TI2V-5B video backbone on large-scale
egocentric and robot videos with bidirectional flow matching. The resulting checkpoint
initializes the Stage 2 video-action model.

Training runs on pre-computed 48-channel Wan2.2 VAE latents and UMT5 text embeddings:

- the first 1 or 2 latent frames are condition frames (optionally noised in latent space);
  the loss is computed on the remaining frames only;
- condition tokens use timestep 0 and denoising tokens use the sampled timestep (TI2V
  per-token timesteps);
- text conditioning is dropped with probability `common.lang_uncond_prob` for
  classifier-free guidance, replacing it with the UMT5 embedding of the empty caption
  (`dataset.empty_lang_embed_path`, created with `scripts/encode_caption.py --caption ''`).

## Layout

```text
train/train_video_pretrain.py    training entry
train/sample_video_pretrain.py   in-training evaluation and visualization
data/video_pretrain/             episode discovery (directory scan or DuckDB index) and loading
models/                          Wan video model wrapper and VAE
third_party/wan/                 vendored Wan2.2 modules (Apache-2.0)
configs/                         example training configs, config template, DeepSpeed configs
scripts/                         launcher
```

## Setup

```bash
pip install -r requirements.txt
pip install "flash-attn>=2.8.3" --no-build-isolation   # recommended for GPU training
```

Download Wan2.2-TI2V-5B and point `model.wan.checkpoint_path` in your config to it.

## Training

- `configs/video_pretrain_example.yaml`: a complete run over a directory tree of pre-computed
  latents and UMT5 embeddings.

Replace the `/path/to/...` placeholders, then launch:

```bash
CONFIG=configs/video_pretrain_example.yaml bash scripts/train_video_pretrain.sh
```

Checkpoints are written to `system.checkpoint_dir/checkpoint_step_<N>`. `--finetune_from
<checkpoint_dir>` loads model weights only; `--resume_from <checkpoint_dir>` restores the full
trainer state, including the scheduler and optimizer-step counter in `trainer_state.pt`.
Older checkpoints without this metadata retain support through `checkpoint_step_<N>` names.
Short videos are trimmed before attention; padding is used only for the masked loss.
Checkpoints are released separately.

## License

Apache License 2.0 (see `LICENSE`). Third-party code and its licenses are listed in `NOTICE`.

训练配置仅保留本包的完整示例；`zero1.json` 和 `zero2.json` 分别为 DeepSpeed ZeRO-1、ZeRO-2 配置，按显存与分布式需求选择。

The public Stage1 weight package uses a flat layout: `model.pt` and `config.yaml`
at the repository root. `model.pt` is a model-only PyTorch state dict; it contains
no optimizer or distributed runtime state. The accompanying YAML is a usage
example. Set `finetune.checkpoint_path`
to the actual local `model.pt` path when using it from another working directory.
