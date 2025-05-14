import torch
from torchvision import transforms
from torch.utils.data import Dataset, default_collate
from PIL import Image
import pickle
from collections import Counter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

class DrivingDataset(Dataset):
    def __init__(self, file_list, augment=False, test=False):
        self.samples = file_list
        self.test = test
        self.cmd2idx = {'forward': 0, 'left': 1, 'right': 2}

        self.img_size = (224, 224)
        self.aux_size = (56, 56)  # Perception head output size
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # --- Image ---
        camera = Image.fromarray(data['camera'].astype(np.uint8))
        camera = self.transform(camera)  # (3, 224, 224)

        # --- Motion ---
        pos_xyh = data['sdc_history_feature'][:, :3]
        vel = np.zeros_like(pos_xyh)
        vel[1:] = pos_xyh[1:] - pos_xyh[:-1]
        acc = np.zeros_like(pos_xyh)
        acc[1:] = vel[1:] - vel[:-1]
        timestep = np.linspace(0, 1, 21).reshape(-1, 1)
        ego_flag = np.ones((21, 1))
        history = np.concatenate([pos_xyh, vel, acc, timestep, ego_flag], axis=1)

        cmd_idx = self.cmd2idx[data['driving_command']]

        # --- Semantic & Depth ---
        sem_raw = data['semantic_label'].astype(np.int64)  # (200, 300)
        sem = torch.from_numpy(sem_raw)
        sem_resized = TF.resize(sem.unsqueeze(0), self.aux_size, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)

        depth = data['depth'].astype(np.float32)  # (200, 300, 1)
        depth = torch.from_numpy(depth).permute(2, 0, 1)  # (1, 200, 300)
        depth = TF.resize(depth, self.aux_size, interpolation=transforms.InterpolationMode.BILINEAR)

        # --- Object Presence Masks (resized to aux resolution) ---
        def resize_mask(class_id):
            mask = (sem == class_id).float().unsqueeze(0)
            return TF.resize(mask, self.aux_size, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)

        car_mask = resize_mask(1)             # CAR
        lane_mask = resize_mask(12)           # LANE_LINE
        traffic_light_mask = resize_mask(9)   # TRAFFIC_LIGHT
        truck_mask = resize_mask(2)           # TRUCK

        semantic_mask = torch.stack([
            car_mask,
            lane_mask,
            traffic_light_mask,
            truck_mask
        ], dim=0)  # (4, 56, 56)

        if self.test:
            return {
                'camera': camera,
                'history': torch.FloatTensor(history),
                'command': torch.tensor(cmd_idx).long(),
                'depth': depth,
                'semantic': sem_resized,
                'car_mask': car_mask,                       # (56, 56)
                'lane_line_mask': lane_mask,                # (56, 56)
                'traffic_light_mask': traffic_light_mask,   # (56, 56)
                'truck_mask': truck_mask,                   # (56, 56)
                'semantic_mask': semantic_mask
            }

        # --- Future Trajectory ---
        future = data['sdc_future_feature'][:, :3]

        return {
            'camera': camera,                           # (3, 224, 224)
            'history': torch.FloatTensor(history),      # (21, 10)
            'future': torch.FloatTensor(future),        # (60, 3)
            'command': torch.tensor(cmd_idx).long(),
            'depth': depth,                             # (1, 56, 56)
            'semantic': sem_resized,                    # (56, 56)
            'car_mask': car_mask,                       # (56, 56)
            'lane_line_mask': lane_mask,                # (56, 56)
            'traffic_light_mask': traffic_light_mask,   # (56, 56)
            'truck_mask': truck_mask,                   # (56, 56)
            'semantic_mask': semantic_mask              # (4, 56, 56)
        }
