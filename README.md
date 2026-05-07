# ================================================================
# README.md  — paste this into: README.md
# ================================================================
# Breast Tumor AI — 3-Stage DCE-MRI Pipeline
 
A complete deep learning pipeline for breast MRI analysis using
3-channel DCE-MRI (Dynamic Contrast-Enhanced MRI) volumes.
 
## Pipeline
 
```
Input: 3-channel DCE-MRI (P1, P2, P3)
         ↓
Stage 1: Tumor Detection (3D ResNet18 + CBAM)
         ↓ if positive
Segmentation: Tumor mask + metrics (MONAI UNet3D + DynUNet ensemble)
         ↓
Stage 3: Benign vs Malignant (3D EfficientNet-B0)
```
 
## Results
 
| Stage | Model | Key Metric | Value |
|-------|-------|-----------|-------|
| Stage 1 | 3D ResNet18 + CBAM (33.4M params) | Test AUC | 0.8765 |
| Segmentation | MONAI UNet3D + DynUNet ensemble | Val Dice | 0.7342 |
| Stage 3 | 3D EfficientNet-B0 (4.7M params) | Test AUC | 0.9200 |
 
## Repository Structure
 
```
Breast_Tumor_AI_Project/
├── src/
│   ├── segmentation_3d/        # MONAI UNet3D model + training
│   ├── dynunet_3d/             # DynUNet model + ensemble
│   └── classification/
│       ├── stage1/             # Tumor detection classifier
│       └── stage3/             # Benign vs malignant classifier
├── notebooks/                  # End-to-end pipeline notebooks
├── models/                     # Trained weights (Git LFS)
│   ├── segmentation_3d/
│   ├── dynunet_3d/
│   ├── classification_stage1/
│   └── classification_stage3/
├── docs/                       # Technical documentation
├── outputs/                    # Generated plots (gitignored)
├── requirements.txt
└── README.md
```
 
## Setup
 
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Breast_Tumor_AI_Project.git
cd Breast_Tumor_AI_Project
 
# Install Git LFS for model weights
git lfs install
git lfs pull
 
# Create environment
python -m venv ml
ml\Scripts\activate          # Windows
pip install -r requirements.txt
```
 
## Dataset
 
- **Existing patients** (IDs 1–233): 1.5T Cartesian breast MRI
- **FastMRI** (IDs 234+): 3T radial GRASP acquisition
- **BreastDx normals** (IDs 234–249): Additional negative cases
- Total: 516 patients (classification), 233 (segmentation)
 
Datasets are not included in this repository.
Place preprocessed `.npy` files in `data/patients_combined/` and
`data/classification_stage{1,3}/` following the structure in
`data/classification_split_summary.txt`.
 
## Hardware
 
Trained on:
- RTX 3050 6GB Laptop (Stage 1 classification)
- RTX 2000 Ada 16GB (Segmentation + Stage 3)
 