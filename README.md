<div align="center">

# Motus2: A Self-Evolving General World Model for Dexterous Manipulation

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?logo=googlechrome&logoColor=white)](https://motus-robotics.github.io/motus2/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.30237-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.30237v2)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/motus-robotics)

</div>

## Overview

General embodied agents should perceive, predict, act, evaluate, and improve within a unified system. World models have shown great promise in building such agents, yet existing models typically append an action output head to a world simulator, without coupling them into a closed decision-and-learning loop for policy improvement. We present **Motus2**, a self-evolving general world model for dexterous manipulation. Motus2 advances world modeling through model scaling and data scaling. For model scaling, a single model with shared weights exposes three control interfaces: a policy (world–action model), a simulator (action-conditioned world model), and an evaluator (value model). The policy proposes candidate action chunks, the simulator predicts their visual consequences, and the evaluator assesses the predicted outcomes. Their coupling forms a closed decision-and-learning loop for policy improvement. This formulation uses curated expert demonstrations for action learning, while failed and suboptimal interactions provide valuable evidence for dynamics modeling and value learning. For data scaling, Motus2 progresses from large-scale monocular egocentric data to synchronized stereo egocentric data, followed by robot-domain adaptation with robot trajectories and supplementary human–robot alignment data. Motus2 further studies global-autoregressive and hybrid-memory extensions of its sliding-window context, adds tactile feedback for contact-aware control, and is instantiated on a fully biomimetic platform with stereo vision, dual arms, dual dexterous hands, and tactile sensing. Together, egocentric data scaling and closed-loop general world model scaling provide a general path toward self-evolving dexterous manipulation.

<p align="center">
  <img src="https://motus-robotics.github.io/motus2/assets/img/latest/motus2_overview.png" alt="Motus2 overview" width="96%">
</p>

## Updates

- **[2026-11]** Released the Stage 1 checkpoint and code for video pretraining, video–action pretraining, supervised fine-tuning, and memory training/inference.

### Open-Source Roadmap

- [x] Stage 1 pretraining checkpoints
- [ ] Stage 2 pretraining checkpoints
- [x] Video pretraining code (Stage 1)
- [x] Video–Action (Value) training code (Stage 2 pretraining and supervised fine-tuning)
- [x] Memory and variable-length training infrastructure

## Model Checkpoints

| Model | Description | Download |
| --- | --- | --- |
| Motus2 Stage 1 | Video-pretrained model for initializing subsequent training stages | [Hugging Face](https://huggingface.co/motus-robotics/Motus2_stage1) |

## Getting Started

```bash
git clone https://github.com/shengshu-ai/Motus2.git
cd Motus2
```

Follow the guide for the training stage you want to use:

| Stage | Guide |
| --- | --- |
| Video pretraining | [Stage 1](stage1/README.md) |
| Video–action pretraining | [Stage 2](stage2/README.md) |
| Supervised fine-tuning and policy inference | [SFT](sft/README.md) |
| Memory training and streaming inference | [Memory](memory/README.md) |

Each guide includes installation, data preparation, and training instructions. Policy server and client examples are included in the Stage 2, SFT, and Memory guides.

## Links

- [Project page](https://motus-robotics.github.io/motus2/)
- [Paper](https://arxiv.org/abs/2608.30237v2)
- [Models](https://huggingface.co/motus-robotics)

## Citation

If you find Motus2 useful in your research, please cite:

```bibtex
@misc{bi2026motus2,
  title         = {Motus2: A Self-Evolving General World Model for Dexterous Manipulation},
  author        = {Hongzhe Bi and Zihao Zhou and Yihang Tang and Jingrui Pang and
                   Shuhe Huang and Haitian Liu and Runqing Wang and Shuai Huang and
                   Yichen Wang and Yiming Cheng and Ruowen Zhao and Zhenghua Li and
                   Hengkai Tan and Xiaolong Liu and Jinhui Wan and Jiabao Liu and
                   Min Zhao and Fan Bao and Jun Zhu},
  year          = {2026},
  eprint        = {2608.30237},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.30237}
}
```
