import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from src.data.texture_maps import process_texture

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
        
        # 정답 값 추출: 이제 roughness(1개)만 사용
        targets = torch.tensor(self.df.iloc[idx]['roughness'], dtype=torch.float32)

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

    
    
class CachedFeatureDataset(Dataset):
    def __init__(self, df, cache_root, split_name):
        self.df = df.reset_index(drop=True)
        self.split_cache = Path(cache_root) / split_name
        self.ids = [str(int(Path(p).stem)) for p in self.df['texture_path'].tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        feat_path = self.split_cache / f"{sid}.pt"
        feat = torch.load(str(feat_path), weights_only=False).float()
        target = torch.tensor(self.df.iloc[idx]['roughness'], dtype=torch.float32)
        return feat, target
    
def _load_roughness_labels(csv_path):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=None)
    if labels_df.shape[1] == 1:
        return {str(i + 1): float(labels_df.iloc[i, 0]) + 50 for i in range(len(labels_df))}

    if 'id' in labels_df.columns and 'roughness' in labels_df.columns:
        return {k: float(v) + 50 for k, v in labels_df.set_index('id')['roughness'].astype(float).to_dict().items()}

    if 'roughness' in labels_df.columns:
        roughness = labels_df['roughness'].astype(float).tolist()
        return {str(i + 1): roughness[i] + 50 for i in range(len(roughness))}

    # return {str(i + 1): float(labels_df.iloc[i, 1]) for i in range(len(labels_df))} # second col (labels_df.iloc[i, 1])
    return {str(i + 1): [v + 50 if isinstance(v, (int, float)) else v for v in labels_df.iloc[i].tolist()] for i in range(len(labels_df))} # use all HA


def _find_image_path(directory, sid, exts):
    for ext in exts:
        path = os.path.join(directory, f"{sid}{ext}")
        if os.path.exists(path):
            return path
    return None


def build_original_dataframe(base_dir="data/original"):
    csv_path = os.path.join(base_dir, "ParticipantData.csv")
    texture_dir = os.path.join(base_dir, 'texture_image')

    label_map = _load_roughness_labels(csv_path)

    texture_files = [
        p for p in Path(texture_dir).iterdir()
        if p.suffix.lower() in {'.png', '.jpg'}
    ]
    texture_files = sorted(texture_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for tex_path in texture_files:
        sid = tex_path.stem
        height_dir, normal_dir = process_texture(tex_path)
        
        # roughness = label_map.get(sid)
        haptic_attribute_list = label_map.get(sid)

        rows.append({
            'texture_path': str(tex_path),
            'normal_path': normal_dir,
            'height_path': height_dir,
            # 'roughness': float(roughness)
            'haptic_attribute': haptic_attribute_list
        })

    return pd.DataFrame(rows)


def build_dataframe(base_dir="data"):
        csv_path = os.path.join(base_dir, "adjective_rating_shuffled.csv")

        # If explicit train/valid id lists exist, use them
        train_ids_path = os.path.join(base_dir, 'train_ids.csv')
        valid_ids_path = os.path.join(base_dir, 'valid_ids.csv')
        if os.path.exists(train_ids_path) and os.path.exists(valid_ids_path):
            # read ids (assume header 'id')
            train_ids = pd.read_csv(train_ids_path)['id'].astype(str).tolist()
            valid_ids = pd.read_csv(valid_ids_path)['id'].astype(str).tolist()

            def build_from_id_list(id_list, split_name=None):
                rows = []
                for sid in id_list:
                    # sid corresponds to image filename without extension
                    # prefer split-specific directories (e.g. data/train/texture_image)
                    if split_name:
                        tex_dir = os.path.join(base_dir, split_name, 'texture_image')
                        nor_dir = os.path.join(base_dir, split_name, 'normal_map')
                        hei_dir = os.path.join(base_dir, split_name, 'height_map')
                    else:
                        tex_dir = os.path.join(base_dir, 'texture_image')
                        nor_dir = os.path.join(base_dir, 'normal_map')
                        hei_dir = os.path.join(base_dir, 'height_map')

                    tex = find_image_path(tex_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    nor = find_image_path(nor_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    hei = find_image_path(hei_dir, int(sid), ['.png', '.jpg', '.JPG'])
                    if all([tex, nor, hei]):
                        # labels CSV may not be present; try to read adjective_rating_shuffled.csv if available
                        if os.path.exists(csv_path):
                            labels_df = pd.read_csv(csv_path, header=None)
                            idx = int(sid) - 1
                            rough = labels_df.iloc[idx][0]
                        else:
                            rough = 0.0
                        rows.append({'texture_path': tex, 'normal_path': nor, 'height_path': hei, 'roughness': rough})
                return pd.DataFrame(rows)

        # build using split subfolders if present
        train_df = build_from_id_list(train_ids, split_name='train')
        valid_df = build_from_id_list(valid_ids, split_name='valid')
        test_df = pd.DataFrame([])
        return train_df, valid_df, test_df


def find_image_path(directory, idx, exts):
    """확장자 순회하며 실제 파일 경로 탐색"""
    for ext in exts:
        path = os.path.join(directory, f"{idx}{ext}")
        if os.path.exists(path):
            return path
    return None