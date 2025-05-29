import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from models.planner import CASPStylePlanner
from data.dataset import DrivingDataset

if __name__ == "__main__":
    # === Load Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASPStylePlanner()
    model.load_state_dict(torch.load("best_model.pth")) # load your model path here
    model.to(device)
    model.eval()

    # === Load Validation Data ===
    val_data_dir = "val_real"
    val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]
    val_dataset = DrivingDataset(val_files, test=False)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)

    # === Accumulators ===
    all_preds, all_futures, all_histories = [], [], []
    all_images = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validating Full Set]"):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            traj_pred = model(
                camera,
                history
            )

            all_preds.append(traj_pred.cpu())
            all_futures.append(future.cpu())
            all_histories.append(history.cpu())
            all_images.append(camera.cpu())
        
    # === Stack all ===
    def stack_all(tensors): return torch.cat(tensors, dim=0)

    all_preds = stack_all(all_preds).numpy()
    all_futures = stack_all(all_futures).numpy()
    all_histories = stack_all(all_histories).numpy()
    all_images = stack_all(all_images).numpy()

    # === Sample k examples ===
    all_ades = []
    for idx in range(len(all_preds)):
        pred_xy = all_preds[idx, :, 0:2]
        future_xy = all_futures[idx, :, 0:2]
        diffs = pred_xy - future_xy
        dist = np.sqrt((diffs**2).sum(axis=-1))
        ade = dist.mean()
        all_ades.append(ade)

    # === Filter examples with ADE > 1.6 ===
    high_error_indices = [idx for idx, ade in enumerate(all_ades) if ade > 1.6]
    print(f"Found {len(high_error_indices)} examples with ADE > 1.6")

    # === Sample k examples from high error cases ===
    k = min(6, len(high_error_indices))
    if len(high_error_indices) >= k:
        selected_indices = random.sample(high_error_indices, k)
    else:
        selected_indices = high_error_indices  # Use all if fewer than k

    # === RGB Images ===
    fig, axis = plt.subplots(1, k, figsize=(4*k, 4))
    for i, idx in enumerate(selected_indices):
        img = np.clip(all_images[idx].transpose(1, 2, 0), 0, 1)
        axis[i].imshow(img)
        axis[i].axis("off")
        axis[i].set_title(f"Sample {idx}")

    # === Trajectories ===
    fig, axis = plt.subplots(1, k, figsize=(4*k, 4))
    for i, idx in enumerate(selected_indices):
        ax = axis[i]
        past_xy = all_histories[idx, :, 0:2]  # (T_past, 2)
        future_xy = all_futures[idx, :, 0:2]  # (T_future, 2)
        pred_xy = all_preds[idx, :, 0:2]      # (T_future, 2)

        # Compute ADE
        diffs = pred_xy - future_xy
        dist = np.sqrt((diffs**2).sum(axis=-1))  # shape: (T_future,)
        ade = dist.mean()  # average displacement error

        # Plot
        ax.plot(past_xy[:, 0], past_xy[:, 1], "o-", color="gold", label="Past")
        ax.plot(future_xy[:, 0], future_xy[:, 1], "o-", color="green", label="Future")
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "o-", color="red", label=f"Pred, ADE={ade:.2f}")
        ax.axis("equal")
        ax.set_title(f"Trajectory {idx}")
        ax.legend()

    plt.tight_layout()
    plt.show()
