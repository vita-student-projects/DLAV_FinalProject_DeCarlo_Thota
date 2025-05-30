import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np

def get_sim_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

def get_real_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

def get_test_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

class DrivingDataset(Dataset):
    def __init__(self, file_list, is_real = False, test=False, validate=False):
        self.samples = file_list
        self.test = test
        self.validate = validate
        self.cmd2idx = {'forward': 0, 'left': 1, 'right': 2}

        if self.test or self.validate:
            self.transform = get_test_transform()
        else:
            self.transform = get_real_transform() if is_real else get_sim_transform()



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        camera = Image.fromarray(data['camera'].astype(np.uint8))
        camera = self.transform(camera)

        pos_xyh = data['sdc_history_feature'][:, :3]
        vel = np.zeros_like(pos_xyh)
        vel[1:] = pos_xyh[1:] - pos_xyh[:-1]
        acc = np.zeros_like(pos_xyh)
        acc[1:] = vel[1:] - vel[:-1]

        timestep = np.linspace(0, 1, 21).reshape(-1, 1)
        ego_flag = np.ones((21, 1))

        history = np.concatenate([
            pos_xyh,          # (21, 3)
            vel,              # (21, 3)
            acc,              # (21, 3)
        ], axis=1)  # (21, 9)

        if not self.test:
            future = data['sdc_future_feature'][:, :3]
            return {
                'camera': camera,
                'history': torch.FloatTensor(history),   # (21, 10)
                'future': torch.FloatTensor(future),     # (60, 3)
            }
        else:
            return {
                'camera': camera,
                'history': torch.FloatTensor(history),
            }
