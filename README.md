# CoatCervix

**A Lightweight Hybrid Transformer for Efficient Cervical Cancer Screening**

---

## Introduction

CoatCervix is a specialized deep learning architecture for automated classification of cervical transformation zone types from colposcopy imagery. Designed for deployment in resource-constrained clinical environments, the model combines the representational power of vision transformers with the computational efficiency required for point-of-care screening applications.

The architecture addresses three fundamental challenges in computer-aided cervical cancer screening:

1. **Diagnostic Accuracy**: Achieving clinician-level classification performance across transformation zone types
2. **Computational Efficiency**: Enabling inference on edge devices without specialized hardware accelerators
3. **Robustness**: Maintaining consistent performance across variable image acquisition conditions

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Methodology](#methodology)
   - [Data Preparation and Stratification](#data-preparation-and-stratification)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Network Architecture](#network-architecture)
   - [Custom Classification Head](#custom-classification-head)
4. [Training Protocol](#training-protocol)
5. [Repository Structure](#repository-structure)
6. [Acknowledgments](#acknowledgments)

---

**Dependencies:**

```
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0
ptflops>=0.6.0
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/username/CoatCervix.git
cd CoatCervix

# Install dependencies
pip install -r requirements.txt

# Execute training pipeline
python main.py
```

---

## Dataset

The refined subset of the Intel & MobileODT Cervical Cancer 
Screening dataset used in this study — comprising 766 manually 
reviewed and preprocessed colposcopic images — is publicly 
available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19212544.svg)](https://doi.org/10.5281/zenodo.19212544)

> Medina-López, R. et al. (2025). *Refined Subset of the Intel 
> & MobileODT Cervical Cancer Screening Dataset: Manual Curation 
> and ROI Preprocessing for Colposcopic Image Classification*. 
> Zenodo. https://doi.org/10.5281/zenodo.19212544

## Methodology

### Data Preparation and Stratification

The experimental dataset derives from the **Intel and MobileODT Cervical Cancer Screening** competition, subjected to rigorous quality control and stratified partitioning.

**Quality Control Protocol:**

| Criterion | Action |
|-----------|--------|
| Motion artifacts | Exclusion of images exhibiting blur from patient movement |
| Illumination defects | Removal of under/overexposed acquisitions |
| Occlusion | Filtering of images with excessive speculum interference |
| Label verification | Expert cross-validation of transformation zone annotations |

**Stratified Data Partitioning:**

The dataset was partitioned using stratified random sampling to preserve class distributions across subsets. The stratification procedure ensures proportional representation of each transformation zone type (Type 1, Type 2, Type 3) within training, validation, and test sets.

| Partition | n | Proportion | Purpose |
|-----------|---|------------|---------|
| Training | 490 | 64% | Model parameter optimization |
| Validation | 122 | 16% | Hyperparameter tuning and early stopping |
| Test | 154 | 20% | Final performance evaluation |
| **Total** | **766** | **100%** | — |

**Reproducibility:** All partitions were generated using a fixed random seed (SEED=42), enabling exact replication of experimental conditions. The stratification guarantees that class imbalance characteristics are consistent across all data subsets.

### Preprocessing Pipeline

Input images undergo a standardized preprocessing pipeline prior to model ingestion:

**Stage 1: Contrast Enhancement via CLAHE**

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied exclusively to the luminance channel (L*) in CIE LAB color space. This approach enhances local contrast while preserving chromatic information critical for tissue differentiation.

Color space transformation: `RGB → Linear RGB → XYZ (D65) → LAB`

| CLAHE Parameter | Value | Justification |
|-----------------|-------|---------------|
| Clip Limit | 2.0 | Constrains contrast amplification to prevent noise enhancement |
| Tile Grid | 8 × 8 | Provides sufficient local adaptivity for colposcopy imagery |

**Stage 2: Normalization**

Post-enhancement normalization using ImageNet channel statistics:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{x}_c=\frac{x_c-\mu_c}{\sigma_c+\epsilon}" alt="Normalization"/>
</p>

where μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225], and ε = 10⁻⁸.

### Network Architecture

CoatCervix employs the **CoaT-Lite Medium** backbone, a hybrid vision transformer implementing serial co-attention mechanisms. The architecture processes input through alternating convolutional and transformer stages:

| Stage | Operation | Output |
|-------|-----------|--------|
| Stem | Convolutional embedding | Patch tokens |
| Stages 1-2 | Convolutional blocks | Local feature hierarchies |
| Stages 3-4 | Transformer blocks with co-scale attention | Global context representations |
| Head | Custom classification module | Class logits |

**Architectural Rationale:**

- **Convolutional stages** establish translation-equivariant local features corresponding to epithelial texture patterns
- **Transformer stages** enable long-range dependency modeling for holistic morphological assessment
- **Co-scale attention** facilitates multi-resolution feature integration without explicit feature pyramid networks

### Custom Classification Head

The backbone output is processed through a task-specific classification head designed for the three-class transformation zone problem:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\text{Linear}(d_{in},\,512)\;\to\;\text{TanhExp}\;\to\;\text{Dropout}(0.3)\;\to\;\text{Linear}(512,\,3)" alt="Head Architecture"/>
</p>

**Component Specifications:**

| Layer | Configuration | Function |
|-------|---------------|----------|
| FC1 | d_in → 512 | Dimensionality reduction from backbone features |
| TanhExp | — | Non-linear activation with regularization properties |
| Dropout | p = 0.3 | Stochastic regularization during training |
| FC2 | 512 → 3 | Final projection to class logits |

**TanhExp Activation Function:**

The intermediate activation employs TanhExp, defined as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\sigma(x)=x\cdot\tanh(e^x)" alt="TanhExp"/>
</p>

with derivative:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\sigma'(x)=\tanh(e^x)+x\cdot%20e^x\cdot\text{sech}^2(e^x)" alt="TanhExp Derivative"/>
</p>

**TanhExp Properties:**

| Property | Mathematical Behavior | Practical Benefit |
|----------|----------------------|-------------------|
| C∞ Continuity | Infinitely differentiable | Stable gradient computation |
| Negative Saturation | σ(x) → 0 as x → -∞ | Implicit L2 regularization effect |
| Positive Linearity | σ(x) ≈ x for x >> 0 | Gradient preservation in deep networks |
| Non-monotonicity | Local minimum at x ≈ -1.28 | Enhanced representational capacity |

---

## Training Protocol

**Execution:**

```bash
python main.py
```

**Hyperparameter Configuration** (defined in `config/settings.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 50 | Maximum training iterations |
| `batch_size` | 16 | Samples per gradient update |
| `lr` | 1×10⁻⁴ | Adam optimizer learning rate |
| `num_classes` | 3 | Output dimensionality |
| `SEED` | 42 | Random state for reproducibility |

**Training Pipeline Stages:**

1. **Initialization**: Backbone loaded with ImageNet pretrained weights; classification head randomly initialized
2. **Data Augmentation**: Standard augmentations applied during training (rotation, flipping, color jitter)
3. **Optimization**: Adam optimizer with default β parameters
4. **Validation**: Per-epoch evaluation computing accuracy and macro-averaged F1-score
5. **Artifact Generation**: Automatic export of training curves, confusion matrices, and Grad-CAM visualizations

---

## Repository Structure

```
CoatCervix/
├── config/
│   └── settings.py              # Hyperparameter definitions
├── data/
│   ├── dataloaders.py           # PyTorch DataLoader factory
│   ├── loaders.py               # Raw data loading utilities
│   ├── preprocessing.py         # CLAHE and normalization
│   └── label_mapper.py          # Label encoding schemes
├── models/
│   ├── activations.py           # TanhExp implementation
│   └── coat_cervix.py           # Model architecture
├── pipeline/
│   └── pipeline_runner.py       # Training orchestration
├── training/
│   ├── evaluate.py              # Evaluation procedures
│   ├── metrics.py               # Metric computation
│   ├── train.py                 # Training loop
│   └── utils.py                 # Utility functions
├── visualization/
│   ├── confusion.py             # Confusion matrix plots
│   ├── gradcam.py               # Grad-CAM generation
│   └── plots.py                 # Training curve visualization
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

---

## Acknowledgments

This research was conducted with support from:

- **Universidad de Caldas**
- **Universidad Autónoma de Manizales**

The authors acknowledge Intel Corporation and MobileODT for providing the foundational cervical cancer screening dataset through the Kaggle competition platform.

---

**Contact:** rafaelmedinalpz@gmail.com
