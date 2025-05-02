import torch
import torch.nn as nn
import torchvision.models as models

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
    def __init__(self, future_len=60, mode_num=1, hidden_dim=256):
        super().__init__()
        self.future_len = future_len
        self.mode_num = mode_num
        self.hidden_dim = hidden_dim

        resnet = models.resnet34(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.image_fc = nn.Linear(512 * 7 * 7, hidden_dim)

        self.history_encoder = TransformerMotionEncoder(input_dim=10, emb_dim=128)
        self.motion_proj = nn.Linear(128, hidden_dim)

        self.cmd_embedding = nn.Embedding(3, 32)
        self.mode_embedding = nn.Embedding(mode_num, 32)

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32 + 32, hidden_dim),
            nn.ReLU()
        )

        self.decoder_gru = nn.GRUCell(hidden_size=hidden_dim, input_size=hidden_dim + 3)
        self.pred_head = nn.Linear(hidden_dim, 5)

        self.scheduled_sampling_prob = 1.0

    def forward(self, camera, history, command, mode_id=None, gt_future=None, decode_len=None, use_scheduled_sampling=True):
        B = camera.size(0)
        device = camera.device
        decode_len = decode_len or self.future_len

        x_img = self.image_encoder(camera)
        x_img = x_img.view(B, -1)
        x_img = self.image_fc(x_img)

        x_hist = self.history_encoder(history)
        x_hist = self.motion_proj(x_hist)

        cmd_embed = self.cmd_embedding(command)

        if mode_id is None:
            mode_id = torch.zeros(B, dtype=torch.long, device=device)
        mode_embed = self.mode_embedding(mode_id)

        fused_feature = self.fusion_fc(torch.cat([x_img, x_hist, cmd_embed, mode_embed], dim=-1))

        outputs = []
        last_pred = torch.zeros(B, 3, device=device)
        h = fused_feature

        scheduled_sampling = (self.training and gt_future is not None and use_scheduled_sampling)
        if scheduled_sampling:
            use_gt = (torch.rand(B, device=device) < self.scheduled_sampling_prob)

        for t in range(decode_len):
            if scheduled_sampling and t > 0:
                step_gt = gt_future[:, t-1, :]
                last_pred = torch.where(use_gt.view(-1, 1), step_gt, last_pred)

            decoder_input = torch.cat([fused_feature, last_pred], dim=-1)
            h = self.decoder_gru(decoder_input, h)
            out = self.pred_head(h)

            delta = out[:, :3]
            last_pred = last_pred + delta

            outputs.append(torch.cat([last_pred, out[:, 3:]], dim=-1).unsqueeze(1))

        return torch.cat(outputs, dim=1)