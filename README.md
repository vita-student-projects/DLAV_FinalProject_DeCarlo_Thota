# DLAV Phase 3 — Sim-to-Real Generalization
**Author**: Giuseppe De Carlo and Sai Avinash Thota

**Course**: Deep Learning for Autonomous Vehicles

**Last Modification**: 29.05.2025

**Milestone 3: Sim-to-Real Generalization**


---

## Overview — Milestone 3

This project implements an Perception Aware deep learning model for the final project of the course DLAV at EPFL in 2025. It is use for predicting future vehicle trajectories.
This phase upgrades the Phase-1 and Phase-2 end-to-end trajectory planner by training to perceive the real scenario.
This repository implements an **end-to-end trajectory planner** designed to generalize from simulation to real-world driving scenarios. Built on top of our Milestone 1 architecture, it focuses on **robust spatial-temporal encoding**, **domain-augmented training**, and **scheduled sampling** to deliver low ADE performance in real-world scenes.

- **ResNet-34 Visual Backbone** (frozen layers 1–3)
- **GRU-based Decoder** with autoregressive delta prediction
- **Motion History Encoder** (via MLP or GRU)
- **Scheduled Sampling** to improve generalization
- **Sim2Real Data Augmentation** using color jitter, affine transform, blur
- **Weighted Multiterm Loss** including velocity, heading, smoothness & curvature

It uses a GRU decoder with Laplace uncertainty modeling and scheduled sampling.

### Key Features

- **ResNet-34 Visual Backbone** (frozen layers 1–3)
- **GRU-based Decoder** with autoregressive delta prediction
- **Motion History Encoder** (via MLP or GRU)
- **Scheduled Sampling** to improve generalization
- **Sim2Real Data Augmentation** using color jitter, affine transform, blur
- **Weighted Multiterm Loss** including velocity, heading, smoothness & curvature


## Model & training method

To reach the tighter target of ADE < 1.80 we extend the Phase 1 planner with perception-aware auxiliary tasks. The resulting model, CASPStylePlanner, is a multi-task network that still predicts a 60-step future trajectory but is now jointly supervised to estimate depth, semantic segmentation, and the presence of critical affordances (cars, lane lines, traffic-lights, trucks). These extra signals shape the latent representation and act as a powerful self-regulariser during training.

### Architecture overview

- **Visual Encoder**: A ResNet-34 pretrained on ImageNet, where all layers are frozen except `layer4`. Output features are passed through an FC projection layer.
- **Motion Encoder**: History of 21 past positions embedded via MLP.
- **Fusion Layer**: Concatenates visual and motion features before decoding.
- **GRU Decoder**: Predicts 60 future waypoints autoregressively from last output.
- **Scheduled Sampling**: During training, gradually replaces GT with model prediction.

### Training configuration

Defined in `loss.py`, it includes:

- **ADE Loss**: L2 loss on predicted vs GT positions
- **Heading Loss**: Angle difference using vector cosine
- **Velocity Loss**: Speed consistency between predicted vs GT velocities
- **Smoothness Loss**: Penalizes abrupt accelerations
- **Curvature Loss**: Controls turning rate variations

### Domain Adaptation
To adapt simulated training to real world testing, we apply the following domain adaptation in `dataset.py`:
- **ColorJitter**
- **GaussianBlur**
- **RandomAffine**
- **Perspective transform**
- **Random Erasing**
  
These are conditionally enabled based on the image domain (real or sim)

### Results for validation

| Metric       | Value |
|--------------|--------|
| ADE (Validation) | ✅ **1.5** |
| FDE (Validation) | ~4.26     	|

With this method we could get an ADE score < 1.8 and reach the task of Milestone 3 with the permitted input.

---

## Project Structure
```bash
DLAV_Phase3/
├── models/
│   ├── planner.py
│   ├── loss.py
│   └── __init__.py
├── data/
│   ├── dataset.py
│   └── __init__.py
├── utils.py
├── train.py
├── visualize_predictions.py
├── infer.py
├── requirements.txt
├── README.md
```
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

To train our model for this third milestone, first set the below configuration inside `train.py` or directly execute with the default values provided. The training is composed of dynamic loss weighting, scheduled sampling and auxiliary perception tasks introduced progressively.

Configuration to run the training:

```bash
# --- Configuration ---
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 25
SAVE_PATH = 'best_model.pth'
```

Run the following script to train the model:

```bash
python train.py
```

Training automatically:

- Logs ADE/Heading error
- Applies scheduled sampling decay
- Apply sim-to-real augmentations
- Performs early stopping based on ADE
- Saves the best model to 'best_model.pth'

## Model prediction visualisation

Once the training is done and we have saved our `best_model.pth`, you can use the following Python script to visualize how the model performs on the validation dataset. You can observe:

- RGB input images
- Past inputs and predicted against future trajectories
- ADE between predicted and future trajectoires

Launch the visualisation:

```bash
python visualize_predictions.py
```

## Inference for submission
Run the following script to generate the submission file for Kaggle:

```bash
python infer.py --model best_model.pth --data data/test --out submission.csv
```

# References

[1] Y. Hu et al., "UniAD: Planning-Oriented Autonomous Driving," *arXiv preprint arXiv:2212.10156*, 2023. [PDF](documents/2212.10156v2.pdf)

[2] L. Chen et al., "End-to-End Autonomous Driving: Challenges and Frontiers," *arXiv:2306.16927*, 2024. [PDF](documents/2306.16927v3.pdf)

[3] H. Yadav et al., "CASPFormer: Trajectory Prediction from BEV Images with Deformable Attention," *arXiv:2409.17790*, 2024. [PDF](documents/2409.17790v1.pdf)

[4] Xiangyu Bai, Yedi Luo, Le Jiang, Aniket Gupta, Pushyami Kaveti, Hanumant Singh, and Sarah Ostadabbas., "Bridging the Domain Gap between Synthetic and Real-World Data for Autonomous Driving," *arXiv:2306.02631*, 2023. [PDF](documents/2306.02631v1.pdf)
