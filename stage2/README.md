# Motus2: Video-Action (Value) Training

Training code for the Motus2 video-action model (Stage 2 pretraining, mid-training and
post-training). A Wan2.2-TI2V-5B video backbone and a lightweight action expert are trained
jointly with chunk-wise autoregressive flow matching over pre-computed VAE latents. An optional
value head predicts task progress from the same sequence.

## Layout

```text
models/    Motus model (models/motus.py), action expert, value head, RoPE, attention masks
train/     train.py (training entry), sample.py (in-training evaluation)
data/      dataset loaders; each loader documents the on-disk format it reads
utils/     action-space conventions, schedulers, checkpoint key conversion
accel/     optional FlashRT kernels (https://github.com/flashrt-project/FlashRT) used by inference; install flash_rt or set FLASHVLA_ROOT, PyTorch fallback otherwise
third_party/wan/   vendored Wan2.2 modules (Apache-2.0)
configs/   example training configs and DeepSpeed configs
scripts/   launcher
```

## Setup

```bash
pip install -r requirements.txt
pip install "flash-attn>=2.8.3" --no-build-isolation   # optional, see requirements.txt
```

Download Wan2.2-TI2V-5B and point `model.wan.checkpoint_path` in your config to it.
Training consumes pre-computed Wan2.2 VAE latents and UMT5 text embeddings; the loaders under
`data/` describe the expected directory / index formats.

## Training

Example configs in `configs/`:

- `pretrain_example.yaml`: Stage 2 pretraining on stereo human / dexterous-hand episodes
  (`qpos20_rel6d`), initialized from a Stage 1 checkpoint, with a source-weighted mixture read
  from `pretrain_sources_example.json`.

Replace the `/path/to/...` placeholders, then launch:

```bash
CONFIG=configs/pretrain_example.yaml RUN_NAME=stage2_pretrain bash scripts/train.sh
```

Outputs go to `<system.checkpoint_dir>/<config stem>/<RUN_NAME>/`. `--finetune <checkpoint_dir>`
loads model weights only; `--resume <checkpoint_dir>` restores the full trainer state.
Set `finetune.checkpoint_path` to the downloaded Stage 1 `model.pt` file when initializing video weights.

Key model options (`model.*` in the config):

- `mode`: `joint` (video and action denoised together), `video_idm` (video first, then actions
  conditioned on it), or `action_only`.
- `action_mode`: `joint` (14-D bimanual joints) or `qpos{20,22}_rel6d` / `relqpos20_rel6d`
  (dexterous hand qpos plus chunk-relative wrist xyz + rot6d).
- `value.enabled`, `value.target` (`progress` or `vlac`), `value.vm_ratio`: value-head training.
- `wm_ratio`: fraction of samples trained as a world model (clean actions, noisy video).

## Inference

The model exposes a streaming interface with a KV cache: `init_kv_cache`, `fill_kv_cache` /
`fill_kv_cache_from_latent`, and `inference_step_streaming`. Call `model.prepare_for_inference()`
once after loading weights to enable the deployment-only accelerations (fused QKV, fused norms,
cached time embeddings and T5 K/V); never call it on a model that will be trained afterwards.

`inference/` contains a machine-independent HTTP policy server built on this interface and a
client example with a RoboTwin-style adapter; see `inference/README.md`:

```bash
python inference/server.py --config inference/config_example.yaml
```

## License

Apache License 2.0 (see `LICENSE`). Third-party code and its licenses are listed in `NOTICE`.
