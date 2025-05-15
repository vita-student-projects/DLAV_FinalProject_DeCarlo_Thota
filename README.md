# DLAV Phase 2 — Perception Aware- Planning
**Author**: Giuseppe De Carlo and Sai Avinash Thota

**Course**: Deep Learning for Autonomous Vehicles

**Last Modification**: 15.05.2025

**Milestone 2: Perception-Aware Planning**



---

## Overview — Milestone 2

This project implements an Perception Aware deep learning model for the final project of the course DLAV at EPFL in 2025. It is use for predicting future vehicle trajectories.
This phase upgrades the Phase-1 end-to-end trajectory planner by training to perceive the depth.
During training the network no longer learns only “where to drive next”, but simultaneously learns how far every pixel in the camera image is.
Adding this auxiliary perception task enriches the visual features, regularises the encoder, and pushes the validation ADE below the 1.60 m target.

- RGB camera input
- Past motion history
- Driving command (left/forward/right)
- Auxiliary Depth Decoder – three up-convolution layers that reconstruct a 56 × 56 dense depth map from the shared visual features.
- Multi-task Training – the model is supervised by a weighted sum of
- Laplace NLL for future (x,y) coordinates
- Heading & smoothness losses (as in Phase 1)
- L1 depth loss for the predicted map (weight λ tuned with Dynamic-Weight-Averaging).

It uses a GRU decoder with Laplace uncertainty modeling and scheduled sampling.

## Model & Training method

To reach the tighter target of ADE < 1.60 we extend the Phase 1 planner with perception-aware auxiliary tasks. The resulting model, CASPStylePlanner, is a multi-task network that still predicts a 60-step future trajectory but is now jointly supervised to estimate depth, semantic segmentation, and the presence of critical affordances (cars, lane lines, traffic-lights, trucks). These extra signals shape the latent representation and act as a powerful self-regulariser during training.

### Architecture Overview

- **Visual Encoder**: A ResNet34 model (pretrained on ImageNet) extracts features from the RGB camera input.
- **Motion History Encoder**: A lightweight Transformer processes the past 21 steps of vehicle motion (`x`, `y`, `heading`, velocity, acceleration) to encode temporal dynamics. The velocity and the acceleration are estimated by the motion of the 21 steps of the vehicle.
- **Command Embedding**: Driving command (`left`, `right`, `forward`) is embedded and fused with other features to help prediction.
- **Feature Fusion**: The outputs of the motion encoder, image encoder and command embedding are merged and passed through a fusion layer.
- **GRU Decoder**: An autoregressive GRU predicts the future trajectory over 60 steps. At each step, the GRU receives the fused features and the last predicted point.
- **Scheduled Sampling**: During training, the model gradually relies less on using ground-truth points for its own predictions to combat exposure bias.
Architecture Overview
This architecture is a multitask deep learning model that performs future trajectory prediction while leveraging perception-based auxiliary tasks (depth estimation, semantic segmentation, and object presence detection). It is trained end-to-end using RGB input images, motion history, and high-level driving commands.

🔹 1. Visual Backbone (Dual ResNet Towers)
Two ResNet-34 encoders are used:

plan_encoder: extracts spatial features for planning.

percep_encoder: feeds auxiliary heads for perception tasks.

Both produce a feature map of shape (B, 512, 7, 7) from the input RGB image.

🔹 2. Motion History Encoder (Transformer)
A custom TransformerMotionEncoder processes ego vehicle history over 21 steps.

Input: (B, 21, 11) → each step includes position, velocity, acceleration, timestep, ego flag.

Output: a 128-dimensional feature, projected to match the planning hidden size.

🔹 3. Command Embedding
A learned embedding layer maps the categorical driving command (left, forward, right) to a 32-dimensional vector.

🔹 4. Auxiliary Perception Heads
These heads operate on the percep_encoder features:

a. Depth Head
A transposed convolutional decoder that upsamples ResNet features to predict a (1, 56, 56) depth map.

b. Semantic Segmentation Head
Similar to the depth head, this decoder outputs a (14, 56, 56) pixel-wise semantic label map over 14 classes.

c. Affordance Heads
A set of binary classifiers (1 per object type: car, truck, lane line, traffic light).

Operate on global ResNet features to predict object presence scores in the scene.

🔹 5. Semantic Mask Encoder (for Fusion)
Ground truth object masks (e.g., car, lane line) are stacked and passed through a lightweight CNN.

Output: a 128-dimensional scene representation summarizing spatial object presence.

🔹 6. Feature Projection and Fusion
The model conditionally fuses planning and auxiliary features based on the training phase:

Always included:

Flattened ResNet planning features → projected to 256.

Transformer-based motion vector (256).

Command embedding (32).

Conditionally included (after epoch 65):

Depth feature projection: AdaptiveAvgPool + Linear → (64,)

Segmentation projection: AdaptiveAvgPool + Linear → (128,)

Semantic mask encoding: CNN + FC → (128,)

➡️ All vectors are concatenated → passed through a fusion MLP → final 256-dimensional planning vector.

🔹 7. GRU-Based Trajectory Decoder
A GRUCell autoregressively predicts the future trajectory over 60 timesteps.

Input at each step: concatenation of the fused feature vector and the last predicted point (x, y, heading).

Output: delta (Δx, Δy, Δθ) added to the previous output to predict the next point.

➡️ Uses scheduled sampling to gradually replace ground truth with model predictions during training.

🔹 8. Multitask Loss Function
The total loss is a weighted combination of:

Loss Type	Component	Description
Planning	Laplace NLL	Main loss for position prediction
Planning	Heading MSE	Aligns predicted vs. true heading
Planning	Velocity MSE	Encourages correct motion dynamics
Planning	Smoothness	Penalizes acceleration spikes
Planning	Curvature	Penalizes sharp turns
Perception	Depth L1	Pixel-wise depth prediction
Perception	Semantic CE	Per-pixel segmentation accuracy
Affordance	Object BCE	Presence prediction for 4 object types
Perception	Lane Line BCE	Binary classification for lane mask

🔹 9. Dynamic Loss Weighting (DWA)
A custom DWAWeightBalancer adjusts loss weights dynamically across epochs.

It adapts to task difficulty by tracking recent loss history and computing a softmax-like score for each task.

This ensures balanced learning across trajectory and perception tasks.

🔹 10. Optimizer & Training Strategy
Optimizer: Adam (lr=1e-4, weight decay=1e-5)

Scheduler: CosineAnnealingWarmRestarts

Gradient Clipping: norm capped at 5.0

Training Phases:

Epoch 0–24: trajectory loss only

Epoch 25–44: trajectory + depth + segmentation

Epoch 45+: full fusion + lane loss + affordance loss

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
| ADE (Validation) | ✅ **1.5** |
| FDE (Validation) | ~4.18     	|

With this method we could get an ADE score < 2 and reach the task of Milestone 1 with the permitted input.

---

## Project Structure
```bash
DLAV_Phase2/
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
Run the following script to generate the submission file for Kaggle:

```bash
python infer.py --model best_model.pth --data data/test --out submission.csv
```
