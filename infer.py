import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import CASPStylePlanner
from data import DrivingDataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load test data ===
    test_data_dir = "test_public_real"
    test_files = sorted(
        [os.path.join(test_data_dir, fn) for fn in os.listdir(test_data_dir) if fn.endswith(".pkl")],
        key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0])
    )
    test_dataset = DrivingDataset(test_files, test=True)
    test_loader = DataLoader(test_dataset, batch_size=250, num_workers=2)

    # === Load Model ===
    model = CASPStylePlanner()
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()

    all_plans = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)

            traj_pred = model(camera, history, command=None)
            all_plans.append(traj_pred.cpu().numpy()[..., :2])  # x, y only

    # === Flatten Results ===
    all_plans = np.concatenate(all_plans, axis=0)  # (N, T, 2)
    total_samples, T, D = all_plans.shape
    pred_xy_flat = all_plans.reshape(total_samples, T * D)

    # === Save CSV ===
    ids = np.arange(total_samples)
    df_xy = pd.DataFrame(pred_xy_flat)
    df_xy.insert(0, "id", ids)

    col_names = ["id"] + [f"{coord}_{t}" for t in range(1, T+1) for coord in ['x', 'y']]
    df_xy.columns = col_names

    df_xy.to_csv("submission_p3.csv", index=False)
    print(f"CSV saved! Shape: {df_xy.shape}")

    # === Sanity Check ===
    x_cols = [c for c in df_xy.columns if c.startswith("x_")]
    y_cols = [c for c in df_xy.columns if c.startswith("y_")]

    max_x, min_x = df_xy[x_cols].max().max(), df_xy[x_cols].min().min()
    max_y, min_y = df_xy[y_cols].max().max(), df_xy[y_cols].min().min()

    print(f" X: [{min_x:.2f}, {max_x:.2f}] | Y: [{min_y:.2f}, {max_y:.2f}]")