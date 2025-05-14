import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import CASPStylePlanner
from data import DrivingDataset


def load_file_list(folder):
    import glob
    return sorted(glob.glob(os.path.join(folder, "*.pkl")))


def run_inference(model_path, test_folder, output_csv, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = CASPStylePlanner()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === Load dataset ===
    test_files = load_file_list(test_folder)
    test_dataset = DrivingDataset(test_files, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="🚀 Running Inference")):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)

            if "depth" in batch and "semantic_mask" in batch:
                depth = batch["depth"].to(device)
                semantic_mask = batch["semantic_mask"].to(device)
                traj_pred = model(
                    camera=camera,
                    history=history,
                    command=command,
                    depth=depth,
                    semantic_mask=semantic_mask,
                    return_aux=False,
                    force_aux=True
                )
            else:
                traj_pred = model(
                    camera=camera,
                    history=history,
                    command=command,
                    return_aux=False,
                    force_aux=False
                )

            pred_xy = traj_pred[..., :2].cpu().numpy()

            # === Format CSV row-by-row
            for i in range(pred_xy.shape[0]):
                sample_id = os.path.basename(test_files[batch_idx * batch_size + i]).replace(".pkl", "")
                for t in range(pred_xy.shape[1]):
                    all_results.append({
                        "id": f"{sample_id}_{t}",
                        "x": pred_xy[i, t, 0],
                        "y": pred_xy[i, t, 1],
                    })

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv} | Total: {len(df)} rows")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pt", help="Path to trained model")
    parser.add_argument("--data", type=str, default="test_public", help="Path to test .pkl folder")
    parser.add_argument("--out", type=str, default="submission_phase2.csv", help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_inference(args.model, args.data, args.out, args.batch_size)