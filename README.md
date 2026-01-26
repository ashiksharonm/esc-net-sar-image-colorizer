# ESC-Net: SAR Image Colorizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashiksharonm/esc-net-sar-image-colorizer/blob/main/sar_novel_v5_colorsar_protocol.ipynb)

A novel deep learning approach for colorizing Synthetic Aperture Radar (SAR) images using the **ColorSAR Protocol** with a custom **ResNet34 + CBAM** architecture.

## 🎯 Overview

This project translates grayscale SAR images into colorized optical-style images by predicting the **ab** (chrominance) channels in the LAB color space. The approach follows the ColorSAR evaluation protocol, which uses the ground truth **L** (Lightness) channel during inference to ensure fair comparison with existing methods.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ESC-Net Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SAR Input (1ch)                                                   │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────┐                                                   │
│   │  Adapt Conv │  7x7, stride 2, 1→64 channels                    │
│   └─────────────┘                                                   │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │              ResNet34 Encoder + CBAM                    │       │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │       │
│   │  │Layer 1  │──▶│Layer 2  │──▶│Layer 3  │──▶│Layer 4  │ │       │
│   │  │ 64ch    │   │ 128ch   │   │ 256ch   │   │ 512ch   │ │       │
│   │  │ +CBAM   │   │ +CBAM   │   │ +CBAM   │   │ +CBAM   │ │       │
│   │  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘ │       │
│   │       │             │             │             │       │       │
│   └───────┼─────────────┼─────────────┼─────────────┼───────┘       │
│           │             │             │             │               │
│   ┌───────┼─────────────┼─────────────┼─────────────┼───────┐       │
│   │       ▼             ▼             ▼             ▼       │       │
│   │              U-Net Style Decoder                        │       │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │       │
│   │  │  Up 1   │◀──│  Up 2   │◀──│  Up 3   │◀──│  Up 4   │ │       │
│   │  │ +Skip   │   │ +Skip   │   │ +Skip   │   │ 512→256 │ │       │
│   │  └────┬────┘   └─────────┘   └─────────┘   └─────────┘ │       │
│   └───────┼─────────────────────────────────────────────────┘       │
│           │                                                         │
│           ▼                                                         │
│   ┌─────────────┐                                                   │
│   │  Output Conv │  1x1 Conv → Tanh → 2 channels (ab)              │
│   └─────────────┘                                                   │
│           │                                                         │
│           ▼                                                         │
│   ab Output (2ch)  ──────┐                                          │
│                          │                                          │
│   GT L Channel ─────────▶├──▶ LAB → RGB Conversion → Colorized     │
│                          │                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### CBAM (Convolutional Block Attention Module)

CBAM is applied at each encoder level to help the model focus on structurally important features (roads, buildings) rather than speckle noise:

```
Input Feature Map
       │
       ▼
┌──────────────┐
│   Channel    │  AdaptiveAvgPool → FC → ReLU → FC → Sigmoid
│  Attention   │  → Element-wise multiply
└──────────────┘
       │
       ▼
┌──────────────┐
│   Spatial    │  AvgPool + MaxPool → Conv → Sigmoid
│  Attention   │  → Element-wise multiply
└──────────────┘
       │
       ▼
Refined Feature Map
```

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Sentinel-1 (SAR) & Sentinel-2 (Optical) |
| **Total Patches** | 64,000 paired images |
| **Resolution** | 128 × 128 pixels |
| **Format** | `.npz` (NumPy compressed) |
| **Train/Val Split** | 90% / 10% |

Each `.npz` file contains:
- `sar_denoised`: Denoised SAR image (grayscale)
- `rgb`: Corresponding optical RGB image

## 🔧 Training Details

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 32 |
| **Loss Function** | MSE (Mean Squared Error) |
| **Color Space** | LAB (predict ab channels) |
| **Epochs** | 50 |
| **Hardware** | NVIDIA A100 GPU |

## 📈 Evaluation Protocol (ColorSAR)

We follow the **ColorSAR** evaluation protocol:
1. Model predicts **ab** channels from SAR input
2. **Ground Truth L** channel is used for reconstruction (not predicted L)
3. LAB → RGB conversion for final output
4. Metrics: **PSNR** and **SSIM** calculated on RGB

This ensures fair comparison with published results (e.g., ColorSAR achieves ~0.94 SSIM using this protocol).

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above and run all cells.

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/ashiksharonm/esc-net-sar-image-colorizer.git
cd esc-net-sar-image-colorizer

# Install dependencies
pip install torch torchvision matplotlib numpy tqdm scikit-image

# Run the notebook
jupyter notebook sar_novel_v5_colorsar_protocol.ipynb
```

## 📁 Project Structure

```
esc-net-sar-image-colorizer/
├── sar_novel_v5_colorsar_protocol.ipynb  # Main training notebook
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```

## 🔬 Key Contributions

1. **Novel CBAM Integration**: We apply CBAM attention at each encoder level to emphasize structural features over speckle noise
2. **ColorSAR Protocol Compliance**: Fair evaluation using ground truth Lightness channel
3. **Lightweight Architecture**: ResNet34 backbone (vs. heavier ResNet50+DenseNet ensembles in ColorSAR)

## 📚 References

1. Schmitt, M., & Hughes, L. H. (2023). *Benchmarking Protocol for SAR Colorization*
2. Woo, S., et al. (2018). *CBAM: Convolutional Block Attention Module*
3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*

## 📄 License

This project is for academic research purposes.

## 👤 Author

**Ashik Sharon M**
- GitHub: [@ashiksharonm](https://github.com/ashiksharonm)
