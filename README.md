# CoatCervix

**A hybrid convolutional and Transformer model for the classification of cervical transformation zone types**

---

## Introduction

CoatCervix classifies colposcopy images into the three transformation zone types (Type 1, Type 2, and Type 3) of the International Federation for Cervical Pathology and Colposcopy. It uses the CoaT-Lite-Medium backbone (Xu et al., 2021), pretrained on ImageNet and used without internal modifications, followed by a classification head with the TanhExp activation, and contrast enhancement with CLAHE.

Across five random seeds, the model reached a mean accuracy of 79.0% ± 2.0% and a mean macro F1-score of 75.1% ± 2.5% on the test set (81.3% and 78.1% in its best run), with a computational cost of 10.41 GFLOPs and 44.32M parameters. These results come from a single public dataset and have not been validated externally; see the paper for the limitations.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Training Protocol](#training-protocol)
6. [Interpretability](#interpretability)
7. [Repository Structure](#repository-structure)
8. [Acknowledgments](#acknowledgments)

---

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0
ptflops>=0.6.0
seaborn
grad-cam
shap
lime
scikit-image
pandas
scipy
```

`shap`, `lime`, `scikit-image`, `pandas`, and `scipy` are needed only for the scripts in `revision/` and `experiments/`. The experiments of the paper were run with timm 1.0.19.

---

## Quick Start

```bash
git clone https://github.com/Mandara2/CoatCervix.git
cd CoatCervix
pip install -r requirements.txt
python main.py
```

Before running, set `BASE_PATH` in `config/settings.py` (it is a placeholder). The data are read from `<BASE_PATH>/Cervical_clean` as `X_train.npy`, `y_train.npy`, `X_test.npy`, and `y_test.npy`, with images in NHWC format and values in [0, 1].

---

## Dataset

The curated subset of the Intel & MobileODT Cervical Cancer Screening dataset used in this study (766 images) is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19212544.svg)](https://doi.org/10.5281/zenodo.19212544)

> Medina-López, R. et al. (2025). *Refined Subset of the Intel & MobileODT Cervical Cancer Screening Dataset: Manual Curation and ROI Preprocessing for Colposcopic Image Classification*. Zenodo. https://doi.org/10.5281/zenodo.19212544

---

## Methodology

### Curation

Each partition of the original competition was curated independently, and no image was moved between them. Images were excluded according to three criteria, defined with respect to the anatomical structures required to assign a transformation zone type:

| Criterion | Exclusion when |
|-----------|----------------|
| Specular reflection | Glare covers the external cervical os or the squamocolumnar junction |
| Occlusion | The speculum, blood, mucus, or the vaginal walls conceal the transformation zone |
| Blur | Defocus or motion prevents the squamocolumnar junction from being distinguished |

For each retained image, the region of interest was cropped manually to the cervix. The transformation zone labels were kept from the original dataset. An MD5 check found no identical files within or between partitions; it cannot exclude distinct images of the same patient.

| Cervix type | Training partition (raw → curated) | First stage test partition (raw → curated) | Total (raw → curated) |
|---|---|---|---|
| Type 1 | 250 → 127 | 87 → 58 | 337 → 185 |
| Type 2 | 781 → 284 | 265 → 124 | 1,046 → 408 |
| Type 3 | 450 → 130 | 160 → 43 | 610 → 173 |
| **Total** | **1,481 → 541** | **512 → 225** | **1,993 → 766** |

### Data splitting

The test set is the curated first stage test partition (225 images). The curated training partition (541 images) is split into training and validation sets with a stratified split of 80% and 20% (`random_state=42`):

| Partition | n | Purpose |
|-----------|---|---------|
| Training | 432 | Weight updates |
| Validation | 109 | Selection of the checkpoint with the highest macro F1-score |
| Test | 225 | Final evaluation |
| **Total** | **766** | |

### Preprocessing (`data/preprocessing.py`)

1. Resize to 224 × 224 pixels.
2. CLAHE on the L channel of the LAB color space (OpenCV; clip limit 2.0, tile grid 8 × 8).
3. Normalization with the channel mean and standard deviation of the resized training images (computed before CLAHE and applied to the filtered images), with ε = 10⁻⁸. ImageNet statistics are not used.

No data augmentation is applied.

### Model (`models/custom_coat.py`)

The backbone is `timm.create_model("coat_lite_medium", pretrained=True, num_classes=0)`, without internal modifications. The classification head is:

```
Linear(d, 512) → TanhExp → Dropout(0.4) → Linear(512, 3)
```

where TanhExp is σ(x) = x · tanh(eˣ) (Liu and Di, 2020), implemented in `models/activations.py`.

---

## Training Protocol

Hyperparameters are defined in `config/settings.py`:

| Parameter | Value |
|-----------|-------|
| `epochs` | 50 |
| `batch_size` | 16 |
| `lr` | 1 × 10⁻⁴ (Adam) |
| `num_classes` | 3 |
| `SEED` | 42 |

The loss is cross entropy. After each epoch, the model is evaluated on the validation set, and the checkpoint with the highest macro F1-score is saved. The evaluation (`training/evaluate.py`, `training/metrics.py`) reloads that checkpoint and reports accuracy and macro averaged precision, recall, and F1-score on the test set, together with the confusion matrix and the ROC curves.

---

## Interpretability

`visualization/gradcam.py` produces the activation maps of the submitted version of the paper: LayerCAM with eigen smoothing at the last convolutional layer of the network, selected automatically. The quantitative evaluation carried out during the revision showed that maps at this layer are not faithful to the model, and the revised paper uses Grad-CAM at the output of the third stage of CoaT, together with SHAP (expected gradients) and LIME. The code of these analyses, including the faithfulness evaluation with NAOPC and the randomization tests, is in `revision/`.

---

## Repository Structure

```
CoatCervix/
├── config/
│   └── settings.py              # Hyperparameters and paths
├── data/
│   ├── dataloaders.py           # PyTorch DataLoaders
│   ├── loaders.py               # Loading of the .npy files and validation split
│   ├── preprocessing.py         # Resize, CLAHE, and normalization
│   └── label_mapper.py          # Binary label modes (not used by the main pipeline)
├── models/
│   ├── activations.py           # TanhExp
│   └── custom_coat.py           # CustomCoaT: backbone and classification head
├── pipeline/
│   └── pipeline_runner.py       # Training and evaluation pipeline
├── training/
│   ├── evaluate.py              # Evaluation on the test set
│   ├── metrics.py               # Metrics
│   ├── train.py                 # Training loop
│   └── utils.py                 # Utilities (includes calculate_efficiency_metrics, not called by the pipeline)
├── visualization/
│   ├── confusion.py             # Confusion matrix
│   ├── gradcam.py               # LayerCAM maps (submitted version)
│   └── plots.py                 # Training curves and ROC curves
├── experiments/                 # Scripts of the results of the paper not produced by main.py (see experiments/README.md)
├── revision/                    # Analyses added during the peer review (see revision/README.md)
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

---

## Acknowledgments

This research was conducted with support from:

- **Universidad de Caldas**
- **Universidad Autónoma de Manizales**

The authors acknowledge Intel Corporation and MobileODT for the Intel & MobileODT Cervical Cancer Screening dataset, released through the Kaggle competition platform.

---

**Contact:** rafaelmedinalpz@gmail.com
