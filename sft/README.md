# Motus2 真机 SFT 后训练

本包提供真机后训练入口，包含 Tianji/Wuji、Sharpa 和 HandsOn loader，以及动作布局检查和 checkpoint 恢复。Stage2 预训练在同级 `stage2/`，MemoryWAM 在 `memory/`。

视频 RoPE 统一使用 Wan 原生分轴频谱。相对动作在 FP32 计算后再转为模型精度。

## 安装与启动

```bash
pip install -r requirements.txt
CONFIG=configs/sft_dexhand_example.yaml RUN_NAME=dexhand_sft bash scripts/train.sh
```

先替换 YAML 的 `/path/to/...` 路径。`finetune.checkpoint_path` 仅加载权重，`resume.checkpoint_path` 恢复优化器及调度器；两者不要同时设置。多机使用 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。日志可通过 shell 重定向保存在自己的目录。

## 样例和数据约定

- `configs/sft_dexhand_example.yaml`：双手 58D、立体 latent 的真机后训练；数据目录格式见配置注释及 `data/tianji_wuji/tianji_wuji_dataset.py`。

HandsOn manifest 是离线转换后的训练索引；每个 episode 的 `episode_manifest.json`、native action/state、VAE latent、语言特征必须满足 loader 要求。`dataset.action_stats_path` 必须是当前训练集、当前动作布局的统计；手部和手腕 XYZ 的 `min_max` 均映射为 `[0,1]`。统计中的 manifest、chunk/stride 元数据会被校验。语言缓存需具有完整状态及其 loader 要求的元数据。

`finetune.require_action_compatibility` 校验基座动作头布局与归一化；`allow_new_state_encoder` 只允许新增状态编码器。真机示例关闭自动评测。

## 检查

```bash
pip install -r requirements-test.txt
pytest -q tests
python train/train.py --help
```

许可见 `LICENSE`、`NOTICE`。
