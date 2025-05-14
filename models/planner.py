import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet18


class TransformerMotionEncoder(nn.Module):
    def __init__(self, input_dim=10, emb_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(2)
        return x


class CASPStylePlanner(nn.Module):
    def __init__(self, hidden_dim=256, future_len=60, use_aux_fusion=False):
        super().__init__()
        self.future_len = future_len
        self.use_aux_fusion = use_aux_fusion

        # === CNN Towers ===
        resnet_plan = resnet34(pretrained=True)
        self.plan_encoder = nn.Sequential(*list(resnet_plan.children())[:-2])  # (B, 512, 7, 7)

        resnet_percep = resnet34(pretrained=True)
        self.percep_encoder = nn.Sequential(*list(resnet_percep.children())[:-2])  # (B, 512, 7, 7)

        self.plan_fc = nn.Linear(512 * 7 * 7, hidden_dim)

        # === Motion & Command ===
        self.motion_encoder = TransformerMotionEncoder(input_dim=11, emb_dim=128)
        self.motion_proj = nn.Linear(128, hidden_dim)
        self.cmd_embedding = nn.Embedding(3, 32)

        # === Auxiliary Heads ===
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(64, 14, 3, padding=1)
        )

        self.semantic_mask_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )

        # === Feature Projections ===
        self.depth_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7, 64),
            nn.ReLU()
        )

        self.seg_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(14 * 7 * 7, 128),
            nn.ReLU()
        )

        # === Attention Heads ===
        self.attn_heads = nn.ModuleDict({
            obj: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            for obj in ['car', 'lane_line', 'traffic_light', 'truck']
        })

        # === Fusion Layer ===
        base_fusion_dim = hidden_dim * 2 + 32
        aux_dim = 64 + 128 + 128  # depth, seg, semantic mask
        self.fusion_input_dim = base_fusion_dim + aux_dim

        self.fusion_fc = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.ReLU()
        )

        # === Decoder ===
        self.decoder_gru = nn.GRUCell(input_size=hidden_dim + 3 + 4, hidden_size=hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, 3)

        self.scheduled_sampling_prob = 1.0

    def forward(self, camera, history, command, depth=None, semantic_mask=None,
                gt_future=None, decode_len=None, return_aux=True, use_aux=False, force_aux=False):
        B = camera.size(0)
        device = camera.device
        decode_len = decode_len or self.future_len

        # === Towers ===
        x_plan_feat = self.plan_encoder(camera)
        x_plan = self.plan_fc(x_plan_feat.view(B, -1))
        x_percep_feat = self.percep_encoder(camera)
        x_hist = self.motion_encoder(history)
        x_hist = self.motion_proj(x_hist)
        x_cmd = self.cmd_embedding(command)

        # === Aux Heads ===
        depth_out = self.depth_head(x_percep_feat)
        seg_out = self.seg_head(x_percep_feat)

        aux_feats = []
        if force_aux or self.use_aux_fusion or use_aux:
            depth_feat = self.depth_proj(depth if depth is not None else depth_out)
            seg_feat = self.seg_proj(seg_out)
            if semantic_mask is not None:
                sem_mask_feat = self.semantic_mask_encoder(semantic_mask)
                aux_feats = [depth_feat, seg_feat, sem_mask_feat]
            else:
                aux_feats = [depth_feat, seg_feat]

        # === Fusion ===
        if aux_feats:
            fusion_input = torch.cat([x_plan, x_hist, x_cmd, *aux_feats], dim=-1)
        else:
            fusion_input = torch.cat([x_plan, x_hist, x_cmd, torch.zeros(B, 320, device=device)], dim=-1)

        fused = self.fusion_fc(fusion_input)

        # === Attention Features (distance prediction) ===
        attn_preds = {k: head(x_percep_feat) for k, head in self.attn_heads.items()}  # (B, 1)
        attn_feat = torch.cat(list(attn_preds.values()), dim=-1)  # (B, 4)
        fused_with_attn = torch.cat([fused, attn_feat], dim=-1)  # (B, hidden_dim + 4)

        # === Decoder ===
        h = fused
        last_pred = torch.zeros(B, 3, device=device)
        outputs = []

        sampling = self.training and gt_future is not None
        if sampling:
            use_gt = (torch.rand(B, device=device) < self.scheduled_sampling_prob)

        for t in range(decode_len):
            if sampling and t > 0:
                last_pred = torch.where(use_gt.view(-1, 1), gt_future[:, t-1, :], last_pred)
            inp = torch.cat([fused_with_attn, last_pred], dim=-1)
            h = self.decoder_gru(inp, h)
            delta = self.pred_head(h)
            last_pred = last_pred + delta
            outputs.append(last_pred.unsqueeze(1))

        traj_out = torch.cat(outputs, dim=1)

        if return_aux:
            return traj_out, depth_out, seg_out, attn_preds
        else:
            return traj_out
