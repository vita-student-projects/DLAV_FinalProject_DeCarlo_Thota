import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PlanningDynamicLaplaceLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'laplace': 1.0,
            'heading': 0.8,
            'velocity': 0.1,
            'smoothness': 0.05,
            'curvature': 0.5,
        }

    def heading_to_vec(self, theta):
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def dynamic_laplace_nll(self, pred, gt, log_b):
        error = torch.abs(pred - gt)
        b = torch.exp(log_b) + 1e-6
        nll = torch.log(2 * b) + error / b
        return nll.mean()

    def forward(self, pred_seq, gt_seq):
        losses = {}
        pred_pos = pred_seq[..., :2]
        gt_pos = gt_seq[..., :2]
        log_b = pred_seq[..., 3:5]

        losses['laplace'] = self.dynamic_laplace_nll(pred_pos, gt_pos, log_b)

        pred_heading_vec = self.heading_to_vec(pred_seq[..., 2])
        gt_heading_vec = self.heading_to_vec(gt_seq[..., 2])
        losses['heading'] = F.mse_loss(pred_heading_vec, gt_heading_vec)

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        gt_vel = gt_pos[:, 1:, :] - gt_pos[:, :-1, :]
        losses['velocity'] = F.mse_loss(pred_vel, gt_vel)

        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        losses['smoothness'] = pred_acc.norm(dim=2).mean()

        pred_heading_diff = pred_seq[:, 1:, 2] - pred_seq[:, :-1, 2]
        gt_heading_diff = gt_seq[:, 1:, 2] - gt_seq[:, :-1, 2]
        losses['curvature'] = F.mse_loss(pred_heading_diff, gt_heading_diff)

        total_loss = sum(self.weights[k] * losses[k] for k in losses)

        return total_loss, losses

class DWAWeightBalancer:
    def __init__(self, loss_names, temp=2.0):
        self.loss_names = loss_names
        self.temp = temp
        self.history = {name: [1.0, 1.0] for name in loss_names}

    def update(self, current_losses):
        updated_weights = {}
        for name in self.loss_names:
            l_prev2, l_prev1 = self.history[name]
            r = l_prev1 / (l_prev2 + 1e-8)
            weight = len(self.loss_names) * math.exp(r / self.temp)
            updated_weights[name] = weight

        total = sum(updated_weights.values())
        updated_weights = {k: v / total for k, v in updated_weights.items()}

        for name in self.loss_names:
            self.history[name][0] = self.history[name][1]
            self.history[name][1] = current_losses.get(name, 1.0)

        return updated_weights
