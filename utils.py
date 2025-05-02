import torch
import matplotlib.pyplot as plt
import yaml

def get_curved_mask(future, threshold=0.02):
    """
    Identify curved trajectories based on average heading delta.
    
    Args:
        future: (B, 60, 3) tensor
        threshold: float, min average heading delta (radians)

    Returns:
        mask: (B,) boolean tensor where True = curved trajectory
    """
    dh = future[:, 1:, 2] - future[:, :-1, 2]  # heading change
    curvature = dh.abs().mean(dim=1)
    return curvature > threshold

def plot_trajectory(pred, gt=None, ax=None, label='Pred', color='blue'):
    """
    Plot predicted vs ground truth trajectory.
    
    Args:
        pred: (60, 2) or (60, 3)
        gt:   (60, 2) or (60, 3)
        ax:   matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    pred_xy = pred[..., :2].cpu().numpy()
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], label=label, color=color, linewidth=2)

    if gt is not None:
        gt_xy = gt[..., :2].cpu().numpy()
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], '--', label='Ground Truth', color='black', alpha=0.6)

    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    return ax

def load_config(path):
    """
    Load a YAML configuration file.
    
    Args:
        path: path to config.yaml

    Returns:
        dict config
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
