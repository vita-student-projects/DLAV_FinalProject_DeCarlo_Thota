# DLAV Phase 1 — End-to-End Trajectory Planner

**Author**: Giuseppe De Carlo and Sai Avinash Thota

**Course**: Deep Learning for Autonomous Vehicles

**Milestone 1: End-to-End Planning**

---

## Overview

This project implements an end-to-end deep learning model for the final project of the course DLAV at EPFL in 2025. It is use for predicting future vehicle trajectories using:

- RGB camera input
- Past motion history
- Driving command (left/forward/right)

It uses a GRU decoder with dynamic Laplace uncertainty modeling and scheduled sampling.

## Model Development & Training Strategy — Milestone 1

To meet the target of ADE < 2.0 using only the allowed inputs (camera, driving command, ego motion history), we designed a compact but expressive end-to-end trajectory planning model.

### Architecture Overview

- **Visual Encoder**: A ResNet34 CNN backbone (pretrained on ImageNet) extracts semantic visual features from the RGB camera input.
- **Motion History Encoder**: A lightweight Transformer processes the past 21 steps of ego vehicle motion (`x`, `y`, `heading`, velocity, acceleration) to encode temporal dynamics.
- **Command Embedding**: High-level driving intent (`left`, `right`, `forward`) is embedded and fused with other features to guide prediction.
- **Feature Fusion**: The outputs of the motion encoder, image encoder, and command embedding are concatenated and passed through a fusion layer.
- **GRU Decoder**: An autoregressive GRU predicts the future trajectory over 60 steps. At each step, the GRU receives the fused features and the last predicted point.
- **Dynamic Laplace Modeling** *(Optional)*: In an enhanced version, the model predicts Laplace scale parameters (`log bₓ`, `log bᵧ`) per timestep, enabling uncertainty-aware loss modeling.
- **Scheduled Sampling**: During training, the model gradually shifts from using ground-truth points to its own predictions to combat exposure bias.

### Training Configuration

- **Input Modalities**: RGB image, driving command, and ego motion history only
- **Trajectory Losses**:
  - **Laplace NLL loss** for spatial prediction
  - **Heading MSE** to align orientation
  - **Velocity & curvature losses** to encourage realism and smoothness
- **Optimization**:
  - Adam optimizer (`lr=1e-4`, `weight_decay=1e-5`)
  - Cosine annealing learning rate schedule
  - Gradient clipping (`max_norm=5.0`) for stability
- **Data Augmentation**: Random affine transforms, color jittering, and resizing applied to images during training

### Results

| Metric       | Value |
|--------------|--------|
| ADE (Validation) | ✅ **1.6** |
| FDE (Validation) | ~5.4       |
| Curved ADE       | ~1.8       |

This approach met the ADE target for Milestone 1 using only the permitted input signals, demonstrating strong trajectory generation performance in both straight and curved scenarios.

---

## Project Structure

DLAV_P1/

├── models/

│   ├── planner.py

│   ├── loss.py

│   └── __init__.py

├── data/

│   ├── dataset.py

│   └── __init__.py

├── utils.py

├── train.py

├── infer.py

├── requirements.txt

├── README.md

## Setup

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Data 
To download and extract the training, validation, and test datasets, run the following script:
```bash
import gdown
import zipfile

# Training data
download_url = "https://drive.google.com/uc?id=1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr"
output_zip = "dlav_train.zip"
gdown.download(download_url, output_zip, quiet=False)
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(".")

# Validation data
download_url = "https://drive.google.com/uc?id=1wtmT_vH9mMUNOwrNOMFP6WFw6e8rbOdu"
output_zip = "dlav_val.zip"
gdown.download(download_url, output_zip, quiet=False)
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(".")

# Public test data
download_url = "https://drive.google.com/uc?id=1G9xGE7s-Ikvvc2-LZTUyuzhWAlNdLTLV"
output_zip = "dlav_test_public.zip"
gdown.download(download_url, output_zip, quiet=False)
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(".")
```

## Training
Train the planner model:

```bash
python train.py
```

Training automatically:

- Logs ADE/FDE/Heading error
- Applies scheduled sampling decay
- Performs early stopping based on ADE
- Saves the best model to best_model.pth

## Inference for Kaggle Submission

```bash
python infer.py --model best_model.pth --data data/test --out submission.csv
```
