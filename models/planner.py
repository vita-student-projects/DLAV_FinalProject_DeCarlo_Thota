import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

class MotionEncoder(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Flatten(),                         # (B, 11*21)
            nn.Linear(input_dim * 21, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        # x: (B, T=21, C=11)
        B, T, C = x.shape
        x = x.transpose(1, 2)              # → (B, C=11, T=21)
        x = self.cnn(x)                    # → (B, 64)
        return self.fc(x)                  # → (B, 128)


    

class RegionalVisionEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # only this one is trainable

        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        # layer4 is fine-tuned

        self.reduce_conv = nn.Conv2d(512, 128, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, 7, 7)

        reduced = self.reduce_conv(x)
        att_logits = self.attention(reduced)

        B, C, H, W = att_logits.shape
        att_map = att_logits.view(B, C, -1)
        att_map = torch.softmax(att_map, dim=-1)
        att_map = att_map.view(B, C, H, W)

        attended = (x * att_map).sum(dim=(2, 3))
        return self.fc(attended)


    
class CASPStylePlanner(nn.Module):
    def __init__(self, future_len=60, mode_num=1, hidden_dim=256):
        super().__init__()
        self.future_len = future_len
        self.mode_num = mode_num
        self.hidden_dim = hidden_dim

        self.image_encoder = RegionalVisionEncoder(out_dim=hidden_dim)

        self.history_encoder = MotionEncoder(input_dim=3, hidden_dim=128)
        self.motion_proj = nn.Linear(128, hidden_dim)

        self.mode_embedding = nn.Embedding(mode_num, 32)

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.decoder_gru = nn.GRUCell(hidden_size=hidden_dim, input_size=hidden_dim + 3)
        self.pred_head = nn.Linear(hidden_dim, 3)

        self.scheduled_sampling_prob = 1.0

    def forward(self, camera, history, gt_future=None, decode_len=None, use_scheduled_sampling=True):
        B = camera.size(0)
        device = camera.device
        decode_len = decode_len or self.future_len

        x_img = self.image_encoder(camera)

        x_hist = self.history_encoder(history)
        x_hist = self.motion_proj(x_hist)

        fused_feature = self.fusion_fc(torch.cat([x_img, x_hist], dim=-1))

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