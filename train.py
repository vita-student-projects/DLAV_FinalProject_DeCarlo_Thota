import os
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from models.loss import PlanningLaplaceLossSequential, DWAWeightBalancer
from utils import get_curved_mask

from models.planner import CASPStylePlanner
from data.dataset import DrivingDataset

def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100,
          save_path='best_model.pt', early_stop_patience=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    planning_loss_fn = PlanningLaplaceLossSequential(fixed_b=1.2)
    aux_dwa = DWAWeightBalancer(['traj', 'depth', 'seg'])

    best_ade = float('inf')
    patience = 0
    λ_lane, λ_attn = 0.3, 0.35  # fixed

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]")
        aux_losses_epoch = {'traj': [], 'depth': [], 'seg': []}

        use_aux = epoch >= 60  # Phase transition for auxiliary fusion

        for batch in pbar:
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            command = batch["command"].to(device)

            depth_gt = batch["depth"].to(device)
            seg_gt = batch["semantic"].to(device)
            lane_mask = batch["lane_line_mask"].to(device)
            semantic_mask = batch["semantic_mask"].to(device)

            # Optional object presence masks
            attn_obj_masks = {
                obj: batch[f"{obj}_mask"].to(device)
                for obj in ['car', 'traffic_light', 'truck', 'lane_line']
            }

            optimizer.zero_grad()
            decode_len = min(60, 20 + epoch * 2)

            # === Forward ===
            out = model(
                camera, history, command,
                semantic_mask=semantic_mask,
                depth=depth_gt,
                gt_future=future,
                decode_len=decode_len,
                return_aux=True,
                force_aux=use_aux
            )

            if use_aux:
                traj_pred, depth_pred, seg_pred, attn_preds = out
            else:
                traj_pred, depth_pred, seg_pred,_ = out
                attn_preds = {}

            # === Primary loss ===
            curved_mask = get_curved_mask(future[:, :decode_len], threshold=0.01)
            traj_loss, loss_dict = planning_loss_fn(traj_pred, future[:, :decode_len], curved_mask=curved_mask)

            # === Aux losses ===
            depth_loss = F.l1_loss(depth_pred, depth_gt)
            seg_loss = F.cross_entropy(seg_pred, seg_gt)
            lane_pred = seg_pred[:, 12]  # class 12 is lane line
            lane_loss = F.binary_cross_entropy_with_logits(lane_pred, lane_mask)

            # === Attention Presence Loss ===
            attn_loss = 0.0
            if use_aux:
                for obj_name, mask in attn_obj_masks.items():
                    has_object = (mask.view(mask.size(0), -1).max(dim=1).values > 0.5).float()
                    pred_logit = attn_preds[obj_name].squeeze(1)
                    attn_loss += F.binary_cross_entropy_with_logits(pred_logit, has_object)

            # === Log aux losses
            aux_losses_epoch['traj'].append(traj_loss.item())
            aux_losses_epoch['depth'].append(depth_loss.item())
            aux_losses_epoch['seg'].append(seg_loss.item())

            # === λ weights via DWA
            if epoch < 1:
                λ_plan, λ_depth, λ_seg = 1.2, 0.2, 0.2
            elif epoch < 80:
                weights = aux_dwa.update({k: np.mean(v) for k, v in aux_losses_epoch.items()})
                λ_plan = 1.2
                λ_depth = weights['depth'] * 0.4
                λ_seg = weights['seg'] * 0.3
            else:
                λ_plan, λ_depth, λ_seg = 1.0, 0.1, 0.1

            # === Total loss
            if epoch < 25:
                total_loss = traj_loss
            elif epoch < 45:
                total_loss = λ_plan * traj_loss + λ_depth * depth_loss + λ_seg * seg_loss
            else:
                total_loss = (
                    λ_plan * traj_loss +
                    λ_depth * depth_loss +
                    λ_seg * seg_loss +
                    λ_lane * lane_loss +
                    λ_attn * attn_loss
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # === Logging
            loss_dict.update({
                "depth": depth_loss.item(),
                "seg": seg_loss.item(),
                "lane": lane_loss.item(),
                "attn": attn_loss.item() if use_aux else 0.0,
                "λ_plan": λ_plan,
                "λ_depth": λ_depth,
                "λ_seg": λ_seg,
                "total": total_loss.item()
            })
            pbar.set_postfix({k: f"{v:.3f}" for k, v in loss_dict.items()})

        # === Scheduled sampling decay
        model.scheduled_sampling_prob *= 0.97
        model.scheduled_sampling_prob = max(0.1, model.scheduled_sampling_prob)

        # === Validation ===
        model.eval()
        ade_scores, fde_scores = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Validation]"):
                camera = batch["camera"].to(device)
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                command = batch["command"].to(device)

                depth = batch.get("depth", None)
                semantic = batch.get("semantic", None)
                semantic_mask = batch.get("semantic_mask", None)

                # Send to device if they exist
                if depth is not None:
                    depth = depth.to(device)
                if semantic is not None:
                    semantic = semantic.to(device)
                if semantic_mask is not None:
                    semantic_mask = semantic_mask.to(device)

                traj_pred = model(
                    camera,
                    history,
                    command,
                    semantic_mask=semantic_mask,
                    depth=depth,
                    return_aux=False,
                    force_aux=use_aux  # Ensure fusion is on if needed
                )
                ade = ((traj_pred[..., :2] - future[..., :2]) ** 2).sum(-1).sqrt().mean()
                fde = ((traj_pred[:, -1, :2] - future[:, -1, :2]) ** 2).sum(-1).sqrt().mean()

                ade_scores.append(ade.item())
                fde_scores.append(fde.item())

        avg_ade = np.mean(ade_scores)
        avg_fde = np.mean(fde_scores)

        scheduler.step(avg_ade)

        print(f"[Epoch {epoch+1}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")
        print(f"λ weights — plan: {λ_plan:.3f}, depth: {λ_depth:.3f}, seg: {λ_seg:.3f}")

        if avg_ade < best_ade:
            best_ade = avg_ade
            patience = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (ADE: {best_ade:.4f})")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break

if __name__ == "__main__":

    # --- Configuration ---
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOP_PATIENCE = 25
    SAVE_PATH = 'best_model.pth'


    train_data_dir = "train"  # Directory containing training data (Change as needed)
    val_data_dir = "val"      # Directory containing validation data (Change as needed)

    train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
    val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

    # Preprocess all futures for stats (optional)
    all_futures = []
    for file in train_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_futures.append(data['sdc_future_feature'][:, :3])  # only x and y
    all_futures = np.concatenate(all_futures, axis=0)

    # Datasets and loaders
    train_dataset = DrivingDataset(train_files, augment=True, test=False)
    val_dataset = DrivingDataset(val_files, augment=False, test=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # Model, optimizer, scheduler
    model = CASPStylePlanner()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        save_path=SAVE_PATH
    )
