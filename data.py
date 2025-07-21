import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class CustomRegressionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 이미지 경로 추출
        texture_path = self.df.iloc[idx]['texture_path']
        normal_path = self.df.iloc[idx]['normal_path']
        height_path = self.df.iloc[idx]['height_path']
        
        # 정답 값 추출
        targets = torch.tensor([
            self.df.iloc[idx]['roughness'],
            self.df.iloc[idx]['stickiness'],
            self.df.iloc[idx]['bumpiness'],
            self.df.iloc[idx]['hardness']
        ], dtype=torch.float32)

        # 이미지 로드
        texture_img = Image.open(texture_path).convert('RGB')
        normal_map = Image.open(normal_path).convert('RGB')
        height_map = Image.open(height_path).convert('RGB')

        # 변환 적용
        if self.transform:
            texture_img = self.transform(texture_img)
            normal_map = self.transform(normal_map)
            height_map = self.transform(height_map)

        return (texture_img, normal_map, height_map), targets

def build_dataframe(base_dir="data"):
    csv_path = os.path.join(base_dir, "adjective_rating_shuffled.csv")
    df = pd.read_csv(csv_path, header=None)
    
    exts = ['.png', '.jpg', '.JPG']
    data = []

    for idx in range(len(df)):
        # 이미지 경로 탐색
        texture_path = _find_image_path(
            os.path.join(base_dir, "texture_image"), 
            idx+1, exts
        )
        normal_path = _find_image_path(
            os.path.join(base_dir, "normal_map"), 
            idx+1, exts
        )
        height_path = _find_image_path(
            os.path.join(base_dir, "height_map"), 
            idx+1, exts
        )

        if all([texture_path, normal_path, height_path]):
            data.append({
                'texture_path': texture_path,
                'normal_path': normal_path,
                'height_path': height_path,
                # header: roughness|stickiness|bumpiness|hardness
                'roughness': df.iloc[idx][0],
                'stickiness': df.iloc[idx][1],
                'bumpiness': df.iloc[idx][2], 
                'hardness': df.iloc[idx][3]
            })
    
    return split_dataframe(pd.DataFrame(data))

def split_dataframe(df, valid_size=0.15, test_size=0.15, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df, valid_df = train_test_split(
        train_df, test_size=valid_size/(1-test_size), random_state=random_state
    )

    return train_df, valid_df, test_df

def _find_image_path(directory, idx, exts):
    """확장자 순회하며 실제 파일 경로 탐색"""
    for ext in exts:
        path = os.path.join(directory, f"{idx}{ext}")
        if os.path.exists(path):
            return path
    return None