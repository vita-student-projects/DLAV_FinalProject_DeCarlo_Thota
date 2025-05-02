
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import CASPStylePlanner, PlanningDynamicLaplaceLoss, DWAWeightBalancer
from data import DrivingDataset
from utils import get_curved_mask

# --- Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
SAVE_PATH = 'best_model.pth'

# --- Load Data ---
def load_file_list(folder):
    import glob
    return sorted(glob.glob(os.path.join(folder, "*.pkl")))

#Prepare datasets and dataloaders

train_files = load_file_list("data/train")   # update if needed
val_files   = load_file_list("data/val")     # update if needed

train_dataset = DrivingDataset(train_files, test=False)
val_dataset   = DrivingDataset(val_files, test=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model / Loss / Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CASPStylePlanner().to(device)
loss_fn = PlanningDynamicLaplaceLoss()
dwa = DWAWeightBalancer(['laplace', 'heading', 'velocity', 'smoothness', 'curvature'])

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- Training Loop ---
best_ade = float('inf')
patience = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}]")

    for batch in pbar:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        command = batch["command"].to(device)

        optimizer.zero_grad()
        decode_len = min(60, 20 + epoch * 2)

        pred = model(camera, history, command, gt_future=future, decode_len=decode_len)
        loss, loss_dict = loss_fn(pred, future[:, :decode_len])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()})
        train_losses.append(loss.item())

    model.scheduled_sampling_prob = max(0.1, model.scheduled_sampling_prob * 0.95)

    # --- Validation ---
    model.eval()
    ade_list, fde_list, heading_list, curved_ade_list, curved_fde_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            command = batch["command"].to(device)

            pred = model(camera, history, command)

            ade = ((pred[..., :2] - future[..., :2])**2).sum(-1).sqrt().mean()
            fde = ((pred[:, -1, :2] - future[:, -1, :2])**2).sum(-1).sqrt().mean()
            heading_err = nn.functional.mse_loss(pred[..., 2], future[..., 2])

            ade_list.append(ade.item())
            fde_list.append(fde.item())
            heading_list.append(heading_err.item())

            curved_mask = get_curved_mask(future)
            if curved_mask.any():
                curved_pred = pred[curved_mask]
                curved_future = future[curved_mask]
                curved_ade = ((curved_pred[..., :2] - curved_future[..., :2])**2).sum(-1).sqrt().mean()
                curved_fde = ((curved_pred[:, -1, :2] - curved_future[:, -1, :2])**2).sum(-1).sqrt().mean()
                curved_ade_list.append(curved_ade.item())
                curved_fde_list.append(curved_fde.item())

    avg_ade = np.mean(ade_list)
    avg_fde = np.mean(fde_list)
    avg_heading = np.mean(heading_list)
    avg_curved_ade = np.mean(curved_ade_list) if curved_ade_list else 0.0
    avg_curved_fde = np.mean(curved_fde_list) if curved_fde_list else 0.0

    scheduler.step(avg_ade)
    loss_weights = dwa.update({
        'laplace': avg_ade,
        'heading': avg_heading,
        'velocity': 0.0,
        'smoothness': 0.0,
        'curvature': 0.0
    })
    loss_fn.weights = loss_weights

    print(f"[Val] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f} | Heading: {avg_heading:.4f} | Curved ADE: {avg_curved_ade:.4f} | Curved FDE: {avg_curved_fde:.4f}")

    if avg_ade < best_ade:
        best_ade = avg_ade
        patience = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved new best model at ADE {best_ade:.4f}")
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break
