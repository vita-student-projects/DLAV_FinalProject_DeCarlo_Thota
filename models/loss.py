import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PlanningDynamicLaplaceLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'ADE': 2.0,
            'heading': 0.4,
            'velocity': 0.05,
            'smoothness': 0.05,
            'curvature': 0.2,
        }

    def heading_to_vec(self, theta):
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def forward(self, pred_seq, gt_seq):
        losses = {}
        pred_pos = pred_seq[..., :2]

        gt_pos = gt_seq[..., :2]

        # Weighted ADE
        B, T, _ = pred_pos.shape
        w = torch.linspace(1.5, 0.5, T//2).tolist() + torch.linspace(0.5, 1.5, T//2).tolist()
        weights = torch.tensor(w, device=pred_pos.device).unsqueeze(0).unsqueeze(2)  # (1, T, 1)
        diff = (pred_pos - gt_pos) ** 2
        weighted_diff = diff * weights
        losses['ADE'] = weighted_diff.mean()

        # Heading
        losses['heading'] = F.mse_loss(pred_seq[..., 2], gt_seq[..., 2])

        # Velocity
        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        gt_vel = gt_pos[:, 1:, :] - pred_pos[:, :-1, :]
        losses['velocity'] = F.mse_loss(pred_vel, gt_vel)

        # Acceleration (smoothness)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        losses['smoothness'] = pred_acc.norm(dim=2).mean()

        # Curvature
        pred_heading_diff = pred_seq[:, 1:, 2] - pred_seq[:, :-1, 2]
        gt_heading_diff = gt_seq[:, 1:, 2] - gt_seq[:, :-1, 2]
        losses['curvature'] = F.mse_loss(pred_heading_diff, gt_heading_diff)

        total_loss = sum(self.weights[k] * losses[k] for k in losses if k in self.weights)
        return total_loss, losses


class DWAWeightBalancer:
    def __init__(self, loss_names, temp=2.0, max_r=50):
        self.loss_names = loss_names
        self.temp = temp
        self.max_r = max_r
        self.history = {name: [1.0, 1.0] for name in loss_names}

    def update(self, current_losses):
        updated_weights = {}
        for name in self.loss_names:
            l_prev2, l_prev1 = self.history[name]
            r = l_prev1 / (l_prev2 + 1e-8)
            r = min(r, self.max_r)
            weight = len(self.loss_names) * math.exp(r / self.temp)
            updated_weights[name] = weight

        total = sum(updated_weights.values())
        updated_weights = {k: v / total for k, v in updated_weights.items()}

        for name in self.loss_names:
            self.history[name][0] = self.history[name][1]
            self.history[name][1] = current_losses.get(name, 1.0)

        return updated_weights
    


