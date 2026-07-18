import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PatchDataset(Dataset): 
    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        texture_path = row["texture_path"]
        texture_image = Image.open(texture_path).convert("L")
        texture_image = self.transform(texture_image)

        height_path = row["height_path"]
        height_map = Image.open(height_path).convert("L")
        height_map = self.transform(height_map)

        normal_path = row["normal_path"]
        normal_map = Image.open(normal_path).convert("RGB")
        normal_map = self.transform(normal_map)
        
        label = np.array([row["roughness"]], dtype=np.float32) # (1,)
        target = torch.tensor(label, dtype=torch.float32) # raw target

        return texture_image, height_map, normal_map, target
        
class PatchFeatureDataset(Dataset):
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
        self.y_min = torch.as_tensor(y_min, dtype=torch.float32)
        self.y_max = torch.as_tensor(y_max, dtype=torch.float32)

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

def dataset_to_numpy(dataset):
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False
    )

    batch = next(iter(loader))

    # FeatureDataset: (x, y)
    if len(batch) == 2:
        x, y = batch
        X = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

    # SeparatedDataset: (texture, height, normal, y)
    elif len(batch) == 4:
        texture, height, normal, y = batch

        texture = texture.detach().cpu().numpy()
        height = height.detach().cpu().numpy()
        normal = normal.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        X = np.concatenate([texture, height, normal], axis=1)

    else:
        raise ValueError(f"Unsupported batch format with length {len(batch)}")

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    return X, y