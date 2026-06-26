<div align="center">

<img src="https://img.shields.io/badge/Deep%20Learning-Medical%20AI-blueviolet?style=for-the-badge&logo=pytorch" />
<img src="https://img.shields.io/badge/DCE--MRI-3D%20Pipeline-informational?style=for-the-badge&logo=databricks" />
<img src="https://img.shields.io/badge/Classification%20AUC-0.9200-success?style=for-the-badge" />
<img src="https://img.shields.io/badge/Mean%20Dice-0.80-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Best%20Dice-0.9582-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia" />

<br><br>

# 🔬 Breast Tumor AI — DCE-MRI Segmentation & Classification Pipeline

**An end-to-end deep learning pipeline for breast MRI tumor segmentation and malignancy classification using 3-channel Dynamic Contrast-Enhanced MRI (DCE-MRI) volumes.**

<br>

[🧠 Models on Hugging Face](https://huggingface.co/B1015/breast-tumor-ai) &nbsp;|&nbsp;
[📁 GitHub Repository](https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification) &nbsp;|&nbsp;
[▶️ Demo Video](#-demo-video) &nbsp;|&nbsp;
[🚀 Quick Start](#-setup--installation)

</div>

---

> ⚠️ **Medical Disclaimer:** This project is a **research prototype only** and is **not intended for real clinical diagnosis**. All results are for academic and demonstration purposes. Do not use this system to make any medical decisions.

---

## 📋 Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [Results](#-results)
- [Visual Results](#-visual-results)
- [Demo Video](#-demo-video)
- [Model Architectures](#-model-architectures)
- [Research Papers](#-research-papers)
- [Dataset](#-dataset)
- [Data Preprocessing & Dataset Preparation](#-data-preprocessing--dataset-preparation)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [Tumor Feature Extraction](#-tumor-feature-extraction)
- [Hardware](#-hardware)

---

## 🧩 Pipeline Overview

Each DCE-MRI volume contains three temporal phases of contrast agent uptake:

| Phase | Name | Timing | Role |
|-------|------|--------|------|
| **P1** | Pre-contrast | Before injection | Baseline tissue signal |
| **P2** | Peak enhancement | ~90 s post-injection | Maximum tumor enhancement |
| **P3** | Delayed / Washout | ~3–5 min post-injection | Kinetic characterisation (Type I/II/III) |

```
┌──────────────────────────────────────────────────────────┐
│         Input: 3-channel DCE-MRI Volume (P1·P2·P3)       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │  SEGMENTATION — Mask Prediction │
         │  MONAI UNet3D  (35% weight)     │
         │  DynUNet / nnU-Net (65% weight) │
         │  Sliding-window · 96³ patches   │
         │  Output: 3D binary tumor mask   │
         └───────────┬─────────────────────┘
                     │
                     ▼
         ┌─────────────────────────────────┐
         │  CLASSIFICATION                 │
         │  3D EfficientNet-B0             │
         │  Input: masked tumor region     │
         │  Output: Benign / Malignant     │
         └─────────────────────────────────┘
```

---

## 📊 Results

| Stage | Model | Parameters | Metric | Score |
|-------|-------|-----------|--------|-------|
| **Segmentation** | MONAI UNet3D + DynUNet Ensemble | — | Mean Dice | **0.80** |
| **Segmentation** | MONAI UNet3D + DynUNet Ensemble | — | Best Dice (Patient 67) | **0.9582** |
| **Classification** | 3D EfficientNet-B0 | 4.7 M | Test AUC | **0.9200** |

<details>
<summary><b>📖 How to interpret these metrics</b></summary>

**Dice Coefficient:** Measures volumetric overlap between the predicted tumor mask and the ground-truth annotation (range 0–1, where 1 = perfect overlap). A **mean Dice of 0.80** across the test set is solid performance for 3D breast MRI segmentation. The **best single-patient Dice of 0.9582** (patient 67, full sliding-window inference) demonstrates the ensemble's capability on well-defined tumors.

**AUC (Area Under the ROC Curve):** Ranges 0–1; a score of 1.0 is a perfect classifier and 0.5 is random chance. An AUC of **0.9200** for benign vs. malignant classification represents strong clinical-grade discriminative ability.

> **Segmentation inference note:** The pipeline uses sliding-window inference on the full 3D volume (96³ patches, 0.5 overlap, Gaussian importance weighting) with a MONAI:DynUNet ensemble blend of 0.35:0.65. This exactly replicates the training evaluation protocol and avoids spatial-cropping artifacts that would otherwise reduce Dice by ~0.05.

</details>

---

## 🖼️ Visual Results

### Classification — Benign vs. Malignant (Test Set)
> Confusion matrix · ROC curve (AUC = 0.9200) · Per-patient malignancy probability bar chart

![Stage 3 Evaluation](outputs/stage3_eval.png)

---

### Segmentation — All 3 DCE Channels + GT Contour
> All three DCE phases shown with ground-truth tumor contour overlay per axial slice.

![Segmentation Channels](outputs/seg_channels_67.png)

---

### Segmentation — Axial Slice Overlay
> P2 (peak enhancement) channel with green GT contour and red predicted mask contour overlaid.

![Segmentation Slices](outputs/seg_slices_67.png)

---

### Segmentation — Max-Intensity Projections (3 Axes)
> Axial · Coronal · Sagittal MIPs with GT (green overlay) and predicted mask (red overlay). Patient 67, Dice = 0.9582.

![Segmentation MIP](outputs/seg_mip_67.png)

---

### Segmentation — 3D Volume Render
> Filled 3D MIP render of predicted tumor volume.

![Segmentation 3D](outputs/seg_3d_67.png)

---

### Segmentation — Feature Panel (Patient 67)
> Geometry · Texture · DCE kinetics · Intensity histogram inside mask · Shape radar chart.

![Feature Panel](outputs/seg_features_67.png)
---

---

## ▶️ Demo Video

> The video below shows the complete Streamlit app running end-to-end: patient upload, segmentation overlay, classification result, and feature panel.

[![Demo Video](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/VZ097UBy5Kw)

## or
![Project Demo](assets/b_tumor_latest.gif)

> **Note on video hosting:** GitHub's standard file limit is 100 MB. To push the demo video via Git LFS:
> ```bash
> git lfs track "*.mp4"
> git add .gitattributes
> git add demo/demo_video.mp4
> git commit -m "Add demo video via LFS"
> git push origin main
> ```

---

## 🏗️ Model Architectures

### Segmentation Ensemble — MONAI UNet3D + DynUNet

```
MONAI UNet3D
  In channels:  3 (P1, P2, P3)
  Channels:     32 → 64 → 128 → 256 → 320
  Strides:      (2, 2, 2, 2)
  Residual:     True   |   Norm: InstanceNorm3D
  Dropout:      0.1    |   Act:  LeakyReLU(0.01)
  Loss:         Tversky (α=0.3, β=0.7) + Focal (γ=2, α=0.95)

DynUNet (nnU-Net style)
  In channels:  3 (P1, P2, P3)
  Kernels:      [[3,3,3] × 5 stages]
  Strides:      [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
  Filters:      32 → 64 → 128 → 256 → 320
  Deep supervision weights: [1.0, 0.5, 0.25, 0.125, 0.0625]
  Norm:         InstanceNorm3D  |  Residual blocks: True
  Loss:         Tversky (α=0.3, β=0.7) + Focal (γ=2, α=0.95)

Ensemble blend:   MONAI 0.35 × pred  +  DynUNet 0.65 × pred
Inference:        Sliding-window · patch 96³ · overlap 0.5 · Gaussian weighting
Post-processing:  Morphological closing → min-size filter (50 voxels) → keep largest component
```

---

### Classification — 3D EfficientNet-B0

```
Input: (B, 3, D, H, W)  — masked tumor region from segmentation
  └─ Stem Conv3d (3×3×3, stride 2)
  └─ MBConv3D blocks (B0 scaling: width×1.0, depth×1.0)
  └─ Head Conv3d → AdaptiveAvgPool3d
  └─ Dropout(0.4) → Linear(1280 → 1) + Sigmoid
Total:  4.7 M parameters
Loss:   BCE with Label Smoothing (ε=0.1)
```

---

## 📚 Research Papers

### Core Architectures Used

| Architecture | Paper | Venue | Link |
|---|---|---|---|
| **U-Net** | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| **nnU-Net / DynUNet** | Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* | Nature Methods 2021 | [DOI:10.1038/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z) |
| **EfficientNet** | Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* | ICML 2019 | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| **MONAI Framework** | Cardoso et al., *MONAI: An open-source framework for deep learning in healthcare* | arXiv 2022 | [arXiv:2211.02701](https://arxiv.org/abs/2211.02701) |

### Relevant Medical Imaging Research

| Topic | Paper | Venue | Link |
|---|---|---|---|
| **DCE-MRI breast segmentation** | Wang et al., *Breast tumor segmentation in DCE-MRI with tumor sensitive synthesis* | IEEE TNNLS 2021 | [DOI](https://doi.org/10.1109/TNNLS.2021.3056238) |
| **DCE kinetic analysis** | Kuhl et al., *Dynamic breast MR imaging: are signal intensity time course data useful for differential diagnosis?* | Radiology 1999 | [DOI](https://doi.org/10.1148/radiology.211.1.r99ap38101) |
| **Tversky loss** | Salehi et al., *Tversky loss function for image segmentation using 3D fully convolutional deep networks* | MICCAI Workshop 2017 | [arXiv:1706.05721](https://arxiv.org/abs/1706.05721) |
| **Focal loss** | Lin et al., *Focal Loss for Dense Object Detection* | ICCV 2017 | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| **Sliding-window inference** | Çiçek et al., *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation* | MICCAI 2016 | [arXiv:1606.06650](https://arxiv.org/abs/1606.06650) |

---

## 🗂️ Dataset

| Source | Patient IDs | Scanner | Notes |
|--------|-------------|---------|-------|
| Primary cohort | 1 – 233 | 1.5 T Cartesian | Core breast MRI cases with GT masks |
| FastMRI | 234+ | 3 T radial GRASP | Additional acceleration cases |
| BreastDx normals | 234 – 249 | — | Extra negative controls |

| Split | Classification | Segmentation |
|-------|---------------|-------------|
| Train | ~70% | ~70% |
| Validation | ~15% | ~15% |
| Test | ~15% | ~15% |

- **Input format:** Preprocessed 3-channel `.npy` volumes of shape `(3, D, H, W)` — one channel per DCE phase.
- **Voxel spacing assumed:** 1.5 mm isotropic.
- **Raw datasets are not included** in this repository. Place preprocessed `.npy` files in `data/patients_combined/` and `data/classification_stage3/` following the structure in `data/classification_split_summary.txt`.

### 🧪 Sample Test Patients

| Folder | Label | Notes |
|--------|-------|-------|
| `sample_dataset/positive/67/` | Positive (tumor) | Best segmentation case — Dice 0.9582; includes `image.npy` + `label.npy` |
| `sample_dataset/negative/234/` | Negative (no tumor) | FastMRI case; `image.npy` only |

Download directly:
```
https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification/tree/main/sample_dataset
```

---

## 🗃️ Data Preprocessing & Dataset Preparation

### Raw Data Format

Raw data was provided as **NIfTI (.nii / .nii.gz)** files. Each patient folder contained multiple DCE phase series (P0–P6), ground-truth tumor masks, and breast masks. Only **phases P1, P2, and P3** were used — corresponding to pre-contrast, peak enhancement, and delayed washout — as these three phases carry the maximum diagnostic signal for tumor segmentation and kinetic characterisation.

---

### Preprocessing Pipeline

**1. Phase Selection & Stacking**
Phases P1, P2, P3 were extracted and stacked channel-wise into a single 4D array of shape `(3, D, H, W)`. Ground-truth masks were loaded as `(1, D, H, W)` binary arrays.

**2. Intensity Normalization**
Each channel was normalized independently using **percentile-based min-max normalization** to handle outlier intensities common in MRI. Values were clipped and rescaled to `[0, 1]` per volume.

**3. Spatial Handling**
- Voxel spacing assumed at **1.5 mm isotropic** across all volumes
- Volumes reflect-padded to a minimum of **96 × 96 × 96** at load time using MONAI `SpatialPadd(mode="reflect")` — avoids zero-border artifacts, critical for new-cohort patients with thin depth axes (~50 slices)
- No global resampling applied; the patch-based training strategy handles variable volume sizes naturally

**4. Output Format**
```
patient_id/
    image.npy   — shape (3, D, H, W), float32   [P1, P2, P3 stacked]
    label.npy   — shape (1, D, H, W), float32   [binary tumor mask]
```

---

### Dataset Splits

Patient-level 70% / 15% / 15% train/val/test split, stratified to maintain label balance. Two cohorts were combined:

- **Primary cohort** (IDs ≤ 100): Original 1.5T patients, deeper volumes (~220 slices depth)
- **Extended cohort** (IDs > 100): FastMRI / BreastDx patients, shallower volumes (~50 slices depth)

New-cohort patients were **upsampled 2× per epoch** via PyTorch `WeightedRandomSampler` to close the geometry gap between the two cohorts.

---

### Training-Time Augmentation

All augmentations applied on-the-fly per patch during training only. Validation and test sets receive reflect-padding only.

**Patch Sampling**
- Patch size: **96 × 96 × 96** voxels · 6 patches per volume per step
- Ratio: **4 tumor-centred : 1 background** via `RandCropByLabelClassesd`

**Spatial Augmentations**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| Random Flip | prob=0.5, all 3 axes | Left-right / superior-inferior symmetry |
| Random Rotate 90° | prob=0.5, up to 3× | Orientation invariance |
| Random Affine | prob=0.3, rotate ±15°, scale ±10% | Mild shape variation |
| 3D Elastic Deformation | prob=0.2, σ∈[3,5], magnitude∈[50,150] | Generalise across cohort geometry differences |

**Intensity Augmentations**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| Random Intensity Shift | offset=0.1, prob=0.4 | Scanner brightness variation |
| Random Intensity Scale | factor=0.1, prob=0.4 | Contrast variation |
| Gaussian Noise | std=0.05, prob=0.3 | Simulate acquisition noise |
| Gaussian Smoothing | σ∈[0.5,1.0], prob=0.2 | Simulate resolution differences |

---

### Class Imbalance Handling

Tumor voxels typically represent less than **0.05%** of the total volume. Three strategies used in combination:

1. **Tumor-biased patch sampling** — `RandCropByLabelClassesd` ratio `[4, 1]` — 80% of patches centred on tumor regions
2. **Output bias correction** — final conv layer bias initialized to `log(0.0005 / 0.9995) ≈ −7.60`, so sigmoid output starts at ~0.0005 matching tumor prevalence, preventing training collapse
3. **Tversky + Focal loss** — `Tversky(α=0.3, β=0.7)` penalises false negatives ~2.3× more than false positives; combined with `Focal(γ=2, α=0.95)` for hard-voxel focus

---

## 📁 Repository Structure

```
Breast_Tumor_Detection-Classification/
│
├── src/
│   ├── segmentation_3d/            # MONAI UNet3D — model definition + training
│   ├── dynunet_3d/                 # DynUNet — model + ensemble
│   └── classification/
│       └── stage3/                 # Malignancy classifier — 3D EfficientNet-B0
│
├── notebooks/
│   ├── full_pipeline.ipynb                     # End-to-end evaluation pipeline
│   └── main_pipeline_v2_no_kinetics.ipynb
│
├── models/                         # Trained weights — stored via Git LFS
│   ├── classification_stage3/
│   │   ├── best_model.pth          # Classification checkpoint (~54 MB)
│   │   └── training_log.json
│   ├── segmentation_3d/
│   │   └── unet3d_best_raw.pth     # MONAI UNet3D checkpoint (~50 MB)
│   └── dynunet_3d/
│       └── dynunet_best_raw.pth    # DynUNet checkpoint (~64 MB)
│
├── sample_dataset/
│   ├── positive/67/                # Sample tumor patient (image.npy + label.npy)
│   └── negative/234/               # Sample normal patient (image.npy)
│
├── outputs/                        # Generated plots and evaluation figures
├── app.py                          # Streamlit frontend app
├── download_models.py              # Auto-downloads weights from Hugging Face Hub
├── requirements.txt                # Full dependency list
├── requirements_app.txt            # Streamlit-only requirements
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10
- CUDA 11.8+ (for GPU inference) — CPU inference also supported but slower
- Git with Git LFS installed

### 1. Clone the Repository

```bash
git clone https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification.git
cd Breast_Tumor_Detection-Classification
```

### 2. Install Git LFS and Pull Model Weights

```bash
git lfs install
git lfs pull
```

> Alternatively, model weights are available on Hugging Face Hub:
> **[huggingface.co/B1015/breast-tumor-ai](https://huggingface.co/B1015/breast-tumor-ai)**

### 3. Create Virtual Environment and Install Dependencies

```bash
# conda (recommended)
conda create -n breast_tumor python=3.10
conda activate breast_tumor

# or venv
python -m venv ml
ml\Scripts\activate         # Windows
source ml/bin/activate      # Linux / macOS

pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import monai; print('PyTorch:', torch.__version__); print('MONAI:', monai.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 🖥️ Running the Streamlit App

```bash
pip install streamlit plotly
streamlit run app.py
```

Opens at `http://localhost:8501`.

### How to Use

**Step 1 —** Accept the disclaimer on the welcome screen.

**Step 2 —** Download a sample patient from GitHub:
```
https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification/tree/main/sample_dataset
```
- `positive/67/` — tumor patient with `image.npy` + `label.npy` (enables GT overlay)
- `negative/234/` — normal patient with `image.npy` only

**Step 3 —** Models auto-download from Hugging Face Hub on first run with progress bars. Cached on subsequent runs.

**Step 4 —** Upload `.npy` file(s) via the sidebar uploader.

**Step 5 —** View results: segmentation overlay on axial slices + MIP views + GT comparison + classification probability + tumor feature panel.

### Models on Hugging Face Hub

| Model | Path | Size |
|-------|------|------|
| Segmentation UNet3D | `models/segmentation_3d/unet3d_best_raw.pth` | ~50 MB |
| Segmentation DynUNet | `models/dynunet_3d/dynunet_best_raw.pth` | ~64 MB |
| Classification | `models/classification_stage3/best_model.pth` | ~54 MB |

---

## 🧮 Tumor Feature Extraction

For each segmented tumor the pipeline computes:

**Geometry**

| Feature | Description |
|---------|-------------|
| Volume | Voxel count × 3.375 mm³ |
| Surface area | Marching-cubes surface × 2.25 mm² |
| Max diameter | Largest bounding-box dimension (mm) |
| Sphericity | `π^(1/3) × (6V)^(2/3) / SA` — 1.0 = perfect sphere |
| Compactness | `V / (SA^(3/2))` |
| Elongation | Ratio of shortest to longest axis |
| T-stage | Estimated from max diameter (T1 ≤ 20 mm · T2 20–50 mm · T3 > 50 mm) |

**DCE Kinetics**

| Feature | Formula |
|---------|---------|
| Enhancement ratio | `(mean_P2 − mean_P1) / |mean_P1|` |
| Washout rate | `(mean_P2 − mean_P3) / |mean_P2|` |
| Kinetic type | Type I Persistent: washout < 0.1 · Type II Plateau: 0.1–0.2 · Type III Washout: > 0.2 |

**Texture (P2 channel inside mask)**

Mean intensity · Std intensity · Min/Max intensity · Contrast `(max − min)` · Homogeneity `1 − std / (max − min)`

---

## ⚙️ Hardware

| Stage | GPU | VRAM | Notes |
|-------|-----|------|-------|
| Segmentation training | RTX 2000 Ada | 16 GB | Sliding-window inference, BF16 AMP |
| Classification training | RTX 3050 Laptop | 6 GB | Mixed-precision (AMP) |
| Inference | Any CUDA GPU | 4 GB+ | Falls back to CPU if no GPU detected |

Mixed-precision inference (`torch.cuda.amp`) is enabled automatically when a CUDA device is detected.

---

## 📜 License

This project is released for academic and research use. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with** PyTorch · MONAI · Streamlit · Hugging Face Hub

<br>

⭐ If this project helped your research, please consider starring the repository.

</div>
