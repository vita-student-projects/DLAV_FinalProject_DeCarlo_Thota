# DLAV Phase 2 — Perception Aware-Planning
**Author**: Giuseppe De Carlo and Sai Avinash Thota

**Course**: Deep Learning for Autonomous Vehicles

**Last Modification**: 16.05.2025

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

## Model & training method

To reach the tighter target of ADE < 1.60 we extend the Phase 1 planner with perception-aware auxiliary tasks. The resulting model, CASPStylePlanner, is a multi-task network that still predicts a 60-step future trajectory but is now jointly supervised to estimate depth, semantic segmentation, and the presence of critical affordances (cars, lane lines, traffic-lights, trucks). These extra signals shape the latent representation and act as a powerful self-regulariser during training.

### Architecture overview

- **Visual Backbone (Dual ResNet Towers)**: Two ResNet-34 models extract features from the RGB input — one feeds the planning branch (`plan_encoder`), and the other powers auxiliary perception heads (`percep_encoder`).
- **Motion History Encoder**: A lightweight Transformer encodes the last 21 steps of ego motion, including estimated velocity and acceleration, outputting a temporal representation.
- **Command Embedding**: Categorical driving intent is embedded into a 32D vector to guide planning decisions.
- **Auxiliary Perception Heads**:
  - **Depth Decoder**: Upsamples ResNet features into a (1, 56, 56) dense depth map.
  - **Semantic Segmentation Decoder**: Predicts (14, 56, 56) segmentation masks.
  - **Affordance Heads**: Binary classifiers predict the presence of objects like cars, lane lines, traffic lights, and trucks.
- **Semantic Mask Encoder**: Processes GT binary masks for 4 semantic categories and encodes them into a compact 128D vector.
- **Feature Fusion**: The planning feature, motion embedding, command embedding, and (optionally) auxiliary features are concatenated and passed through a fusion MLP.
- **GRU Decoder**: An autoregressive GRU predicts the 60-step trajectory by recursively applying delta prediction to the previous output.
- **Scheduled Sampling**: Ground truth is gradually replaced with model predictions to improve stability.
- **Multitask Loss**: Combines Laplace NLL, heading, velocity, curvature, depth (L1), segmentation (CE), affordance (BCE), and lane (BCE).
- **Dynamic Loss Weighting (DWA)**: Learns task weights automatically across training epochs based on recent loss trends.

### Training configuration

- **Input data**: RGB image, driving command, motion history, depth amd semantic label .
- **Trajectory Losses**:
  - **Laplace NLL loss** for spatial prediction
  - **Heading MSE** to adjust orientation
  - **Velocity & curvature losses** to avoid abnormal behaviour between steps.
- **Optimization**:
  - Adam optimizer (`lr=1e-4`, `weight_decay=1e-5`)
  - Cosine annealing learning rate schedule
  - Gradient clipping (`max_norm=5.0`) for stability
- **Data Augmentation**: Random affine transforms, color jittering, and resizing applied to images only during training.

- **Growing auxiliary supervision**:
	- Epochs 0 to 24: Loss depending only on the trajectory (Laplace NLL).
	- Epochs 25 to 44: Add depth and semantic segmentation losses.
	- Epochs 45+: Add binary presence classification for important objects (cars, trucks, traffic lights, lanes) and a lane-line segmentation.



### Results for validation

| Metric       | Value |
|--------------|--------|
| ADE (Validation) | ✅ **1.5** |
| FDE (Validation) | ~4.18     	|

With this method we could get an ADE score < 1.6 and reach the task of Milestone 2 with the permitted input.

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

To train our model for this second milestone, first set the below configuration inside `train.py` or directly execute with the default values provided. The training is composed of dynamic loss weighting, scheduled sampling and auxiliary perception tasks introduced progressively.

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

- Logs ADE/FDE/Heading error
- Applies scheduled sampling decay
- Performs early stopping based on ADE
- Saves the best model to 'best_model.pth'

## Model prediction visualisation

Once the training is done and we have saved our `best_model.pth`, you can use the following Python script to visualize how the model performs on the validation dataset. You can observe:

- RGB input images
- Past inputs and predicted against future trajectories
- Ground truth against predicted depth
- Semantic segmentation maps

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
