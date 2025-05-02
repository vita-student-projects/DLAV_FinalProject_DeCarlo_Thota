import os
import torch
from torch.utils.data import DataLoader
from models import CASPStylePlanner
from data import DrivingDataset
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_file_list(folder):
    import glob
    return sorted(glob.glob(os.path.join(folder, "*.pkl")))

def run_inference(model_path, test_folder, output_csv, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CASPStylePlanner()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test set
    test_files = load_file_list(test_folder)
    test_dataset = DrivingDataset(test_files, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running Inference")):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)

            pred = model(camera, history, command)  # (B, 60, 5)

            pred_xy = pred[..., :2].cpu().numpy()  # Only x, y

            for i in range(pred_xy.shape[0]):
                sample_id = os.path.basename(test_files[batch_idx * batch_size + i]).replace(".pkl", "")
                for t in range(60):
                    x, y = pred_xy[i, t]
                    all_results.append({
                        "id": f"{sample_id}_{t}",
                        "x": x,
                        "y": y
                    })

    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pth")
    parser.add_argument("--data", type=str, default="data/test")
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    run_inference(args.model, args.data, args.out)
