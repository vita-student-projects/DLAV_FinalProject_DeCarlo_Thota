import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np

class DrivingDataset(Dataset):
    def __init__(self, file_list, test=False):
        self.samples = file_list
        self.test = test
        self.cmd2idx = {'forward': 0, 'left': 1, 'right': 2} #Mappping commands to integer labels

        if self.test:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        #return the number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)
        #convert the camera image into PIL and apply transformations
        camera = Image.fromarray(data['camera'].astype(np.uint8))
        camera = self.transform(camera)
        #Extract position, computer velocity and acceleration from history files
        pos_xyh = data['sdc_history_feature'][:, :3]
        vel = np.zeros_like(pos_xyh)
        vel[1:] = pos_xyh[1:] - pos_xyh[:-1]
        acc = np.zeros_like(pos_xyh)
        acc[1:] = vel[1:] - vel[:-1]
        #Add extra features : time and ego vehicle flag
        timestep = np.linspace(0, 1, 21).reshape(-1, 1)
        ego_flag = np.ones((21, 1))
        #Final history feature by concatenating all calculated inputs
        history = np.concatenate([pos_xyh, vel, acc, timestep, ego_flag], axis=1)  # (21, 10)
        command = self.cmd2idx[data['driving_command']]

        if not self.test:
            future = data['sdc_future_feature'][:, :3]
            return {
                'camera': camera,
                'history': torch.FloatTensor(history),   # (21, 10)
                'future': torch.FloatTensor(future),     # (60, 3)
                'command': torch.tensor(command).long()  # scalar
            }
        else:
            return {
                'camera': camera,
                'history': torch.FloatTensor(history),
                'command': torch.tensor(command).long()
            }
