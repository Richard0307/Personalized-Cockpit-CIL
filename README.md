## Privacy-Preserving Continual Learning for Intelligent Cockpit
                  

## Introduction
This project aims to address how in-vehicle AI models can continuously learn the personalized behavioral habits of vehicle owners after leaving the factory while protecting user privacy, and overcome catastrophic forgetting.

## 🚀 Features
- **On-device OOD Detection:** 利用分布外检测识别未知的驾驶员行为。
- **Parameter-Efficient CIL:** 基于预训练 Vision Transformer (ViT) 和 Adapter 的轻量化增量更新。
- **Privacy-Preserving:** 无需云端传输，纯本地 Replay Buffer 经验回放。

## 📂 Project Structure
```text
├── data/           # Dataset and local replay buffer (Ignored in Git)
├── models/         # Network architectures (e.g., ViT+Adapter)
├── utils/          # OOD detection, Herding algorithm, metrics
├── scripts/        # Bash scripts for reproduction
├── trainer.py      # Core CIL training loop with Distillation Loss
├── main.py         # Entry point
└── README.md
