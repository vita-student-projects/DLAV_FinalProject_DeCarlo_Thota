import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models.planner import CASPStylePlanner
from data.dataset import DrivingDataset
from train import train

if __name__ == "__main__":
    train_data_dir = "train"
    val_data_dir = "val"

    train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
    val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

    # Preprocess all futures for stats (optional)
    all_futures = []
    for file in train_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_futures.append(data['sdc_future_feature'][:, :3])  # only x, y
    all_futures = np.concatenate(all_futures, axis=0)

    # Datasets and loaders
    train_dataset = DrivingDataset(train_files, augment=True, test=False)
    val_dataset = DrivingDataset(val_files, augment=False, test=False)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)

    # Model, optimizer, scheduler
    model = CASPStylePlanner()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=200,
        save_path='best_model.pt'
    )
