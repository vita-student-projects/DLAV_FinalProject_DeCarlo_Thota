# DLAV Phase 1 — End-to-End Trajectory Planner

**Author**: Giuseppe De Carlo and Sai Avinash Thota

**Course**: Deep Learning for Autonomous Vehicles

**Last Modification**: 02.05.2025

**Milestone 1: End-to-End Planning**



---

## Overview — Milestone 1

This project implements an end-to-end deep learning model for the final project of the course DLAV at EPFL in 2025. It is use for predicting future vehicle trajectories using:

- RGB camera input
- Past motion history
- Driving command (left/forward/right)

It uses a GRU decoder with Laplace uncertainty modeling and scheduled sampling.

## Model & Training method

To get the ADE < 2.0 using only the inputs (camera, driving command, motion history), we designed an end-to-end trajectory planning model.

### Architecture Overview

- **Visual Encoder**: A ResNet34 model (pretrained on ImageNet) extracts features from the RGB camera input.
- **Motion History Encoder**: A lightweight Transformer processes the past 21 steps of vehicle motion (`x`, `y`, `heading`, velocity, acceleration) to encode temporal dynamics. The velocity and the acceleration are estimated by the motion of the 21 steps of the vehicle.
- **Command Embedding**: Driving command (`left`, `right`, `forward`) is embedded and fused with other features to help prediction.
- **Feature Fusion**: The outputs of the motion encoder, image encoder and command embedding are merged and passed through a fusion layer.
- **GRU Decoder**: An autoregressive GRU predicts the future trajectory over 60 steps. At each step, the GRU receives the fused features and the last predicted point.
- **Scheduled Sampling**: During training, the model gradually relies less on using ground-truth points for its own predictions to combat exposure bias.

### Training Configuration

- **Input data**: RGB image, driving command, and motion history only
- **Trajectory Losses**:
  - **Laplace NLL loss** for spatial prediction
  - **Heading MSE** to adjust orientation
  - **Velocity & curvature losses** to avoid abnormal behaviour between steps.
- **Optimization**:
  - Adam optimizer (`lr=1e-4`, `weight_decay=1e-5`)
  - Cosine annealing learning rate schedule
  - Gradient clipping (`max_norm=5.0`) for stability
- **Data Augmentation**: Random affine transforms, color jittering, and resizing applied to images only during training

### Results for validation

| Metric       | Value |
|--------------|--------|
| ADE (Validation) | ✅ **1.6** |
| FDE (Validation) | ~5.4       |

With this method we could get an ADE score < 2 and reach the task of Milestone 1 with the permitted input.

---

## Project Structure

DLAV_Phase1/

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
To download and extract the training, validation, and test datasets for the Milestone 1, run the following script:
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
Configuration to run the training:

```bash
# --- Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
SAVE_PATH = 'best_model.pth'
```

Run the following script to train the model:

```bash
python train.py
```

Training automatically:

- Logs ADE/FDE/Heading error
- Applies scheduled sampling decay
- Performs early stopping based on ADE
- Saves the best model to 'best_model.pth'

## Inference for Submission

```bash
python infer.py --model best_model.pth --data data/test --out submission.csv
```
