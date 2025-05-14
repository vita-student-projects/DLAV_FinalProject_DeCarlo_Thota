import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from models.planner import CASPStylePlanner
from data.dataset import DrivingDataset

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CASPStylePlanner()
model.load_state_dict(torch.load("best_model.pt")) # load your model path here
model.to(device)
model.eval()

# === Load Validation Data ===
val_data_dir = "val"
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]
val_dataset = DrivingDataset(val_files, augment=False, test=False)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)

# === Accumulators ===
all_preds, all_futures, all_histories = [], [], []
all_images, all_depths, all_pred_depths = [], [], []
all_semantics = []
all_lane_masks = []
all_attn_preds = {k: [] for k in model.attn_heads.keys()}

with torch.no_grad():
    for batch in tqdm(val_loader, desc="[Validating Full Set]"):
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        command = batch["command"].to(device)
        depth = batch["depth"].to(device)
        semantic = batch["semantic"].to(device)
        lane_mask = batch["lane_line_mask"].to(device)
        sem_mask = batch["semantic_mask"].to(device)

        traj_pred, pred_depth, _, attn_preds = model(
            camera,
            history,
            command,
            depth=depth,
            semantic_mask=sem_mask,
            return_aux=True,
            force_aux=True
        )

        all_preds.append(traj_pred.cpu())
        all_futures.append(future.cpu())
        all_histories.append(history.cpu())
        all_images.append(camera.cpu())
        all_depths.append(depth.cpu())
        all_pred_depths.append(pred_depth.cpu())
        all_semantics.append(semantic.cpu())
        all_lane_masks.append(lane_mask.cpu())
        for k in attn_preds:
            all_attn_preds[k].append(attn_preds[k].cpu())

# === Stack all ===
def stack_all(tensors): return torch.cat(tensors, dim=0)

all_preds = stack_all(all_preds).numpy()
all_futures = stack_all(all_futures).numpy()
all_histories = stack_all(all_histories).numpy()
all_images = stack_all(all_images).numpy()
all_depths = stack_all(all_depths).numpy()
all_pred_depths = stack_all(all_pred_depths).numpy()
all_semantics = stack_all(all_semantics).numpy()
all_lane_masks = stack_all(all_lane_masks).numpy()
all_attn_preds = {k: stack_all(v).numpy() for k, v in all_attn_preds.items()}

# === Sample k examples ===
k = 6
selected_indices = random.sample(range(len(all_preds)), k)

# === RGB Images ===
fig, axis = plt.subplots(1, k, figsize=(4*k, 4))
for i, idx in enumerate(selected_indices):
    img = np.clip(all_images[idx].transpose(1, 2, 0), 0, 1)
    axis[i].imshow(img)
    axis[i].axis("off")
    axis[i].set_title(f"Sample {idx}")
plt.tight_layout()
plt.show()

# === Trajectories ===
fig, axis = plt.subplots(1, k, figsize=(4*k, 4))
for i, idx in enumerate(selected_indices):
    ax = axis[i]
    ax.plot(all_histories[idx, :, 0], all_histories[idx, :, 1], "o-", color="gold", label="Past")
    ax.plot(all_futures[idx, :, 0], all_futures[idx, :, 1], "o-", color="green", label="Future")
    ax.plot(all_preds[idx, :, 0], all_preds[idx, :, 1], "o-", color="red", label="Pred")
    ax.axis("equal")
    ax.set_title(f"Trajectory {idx}")
    ax.legend()
plt.tight_layout()
plt.show()

# === Depth Map ===
vmin = np.percentile(all_depths, 1)
vmax = np.percentile(all_depths, 99)

fig, ax = plt.subplots(2, k, figsize=(4 * k, 6))
for i, idx in enumerate(selected_indices):
    ax[0, i].imshow(all_depths[idx][0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0, i].set_title("GT Depth", pad=10)
    ax[0, i].axis("off")

    ax[1, i].imshow(all_pred_depths[idx][0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1, i].set_title("Pred Depth", pad=10)
    ax[1, i].axis("off")
plt.tight_layout()
plt.show()

# === Semantic and Lane ===
for i, idx in enumerate(selected_indices):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(all_semantics[idx], cmap='tab20')
    axs[0].set_title(f"Semantic Map {idx}")
    axs[0].axis("off")

    axs[1].imshow(all_lane_masks[idx], cmap='gray')
    axs[1].set_title(f"Lane Mask {idx}")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
