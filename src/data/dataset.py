import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class OriginalDataset(Dataset): 
    def __init__(self, df, conf, target_col):
        self.df = df.reset_index(drop=True).copy()
        self.conf = conf
        self.target_col = target_col

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        texture_path = row["texture_path"]
        texture_image = Image.open(texture_path).convert("L")
        texture_image = self.transform(texture_image)

        if self.conf['dataset_input'] == 'texture_maps':
            height_path = row["height_path"]
            height_map = Image.open(height_path).convert("L")
            height_map = self.transform(height_map)

            normal_path = row["normal_path"]
            normal_map = Image.open(normal_path).convert("L")
            normal_map = self.transform(normal_map)
        
        if self.target_col == "haptic_attribute": # (4,)
            label = np.array(row[self.target_col], dtype=np.float32)
        elif self.target_col == "roughness":
            label = np.array([row[self.target_col]], dtype=np.float32) # (1,)

        target = torch.tensor(label, dtype=torch.float32) # raw target

        if self.conf['dataset_input'] == 'texture_maps':
            return texture_image, height_map, normal_map, target    
        elif self.conf['dataset_input'] == 'texture_image':
            return texture_image, target

class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) # raw target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class NormalizedSubset(Dataset):
    def __init__(self, base_dataset, indices, y_min, y_max):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices)
        self.y_min = torch.tensor(y_min, dtype=torch.float32)
        self.y_max = torch.tensor(y_max, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.base_dataset[real_idx]

        if not isinstance(sample, (tuple, list)):
            raise TypeError(f"Expected tuple/list from base_dataset, got {type(sample)}")

        *features, target = sample
        target = (target - self.y_min) / (self.y_max - self.y_min + 1e-8)

        if len(features) == 1:
            return features[0], target
        return (*features, target)