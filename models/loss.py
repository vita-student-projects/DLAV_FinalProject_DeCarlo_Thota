import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PlanningLaplaceLossSequential(nn.Module):
    def __init__(self, fixed_b=1.2, weights=None):
        super().__init__()
        self.fixed_b = fixed_b

        self.loss_names = ['laplace', 'heading', 'velocity', 'smoothness', 'curvature']
        self.weights = weights or {
            'laplace': 1.2,
            'heading': 0.8,
            'velocity': 0.1,
            'smoothness': 0.05,
            'curvature': 0.5,
        }

    def heading_to_vec(self, theta):
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def laplace_nll(self, pred, gt, scale):
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=pred.device, dtype=pred.dtype)
        error = torch.abs(pred - gt)
        return torch.mean(torch.log(2 * scale) + error / scale)

    def forward(self, pred_seq, gt_seq, update_dwa=False, curved_mask=None):
        """
        Args:
            pred_seq: (B, 60, 3)
            gt_seq: (B, 60, 3)
            update_dwa: bool — call with True *once per epoch* to update weights
        """
        losses = {}

        # Position NLL
        losses['laplace'] = self.laplace_nll(pred_seq[..., :2], gt_seq[..., :2], self.fixed_b)

        # Heading direction
        pred_vec = self.heading_to_vec(pred_seq[..., 2])
        gt_vec = self.heading_to_vec(gt_seq[..., 2])
        losses['heading'] = F.mse_loss(pred_vec, gt_vec)

        # Velocity and smoothness
        pred_vel = pred_seq[:, 1:, :2] - pred_seq[:, :-1, :2]
        gt_vel = gt_seq[:, 1:, :2] - gt_seq[:, :-1, :2]
        losses['velocity'] = F.mse_loss(pred_vel, gt_vel)

        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        losses['smoothness'] = pred_acc.norm(dim=2).mean()

        # Curvature (heading rate)
        pred_dtheta = pred_seq[:, 1:, 2] - pred_seq[:, :-1, 2]
        gt_dtheta = gt_seq[:, 1:, 2] - gt_seq[:, :-1, 2]
        losses['curvature'] = F.mse_loss(pred_dtheta, gt_dtheta)

        total = sum(self.weights[k] * losses[k] for k in self.loss_names)
        return total, losses

class DWAWeightBalancer:
    def __init__(self, loss_names, temp=1.0):
        self.loss_names = loss_names
        self.temp = temp
        self.history = {name: [1.0, 1.0] for name in loss_names}  # [t-2, t-1]

    def update(self, current_losses):
        updated_weights = {}

        for name in self.loss_names:
            l_prev2, l_prev1 = self.history[name]
            ratio = torch.tensor(l_prev1 / (l_prev2 + 1e-8))  # avoid div by 0
            weight = len(self.loss_names) * torch.exp(ratio / self.temp)
            updated_weights[name] = weight.item()

        # Normalize to sum to 1
        total = sum(updated_weights.values())
        normalized = {k: v / total for k, v in updated_weights.items()}

        # Update history for next epoch
        for name in self.loss_names:
            self.history[name][0] = self.history[name][1]
            self.history[name][1] = current_losses.get(name, 1.0)

        return normalized
