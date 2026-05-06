import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from pathlib import Path
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

def build_dataframe(base_dir="data"):
    csv_path = os.path.join(base_dir, 'original', 'adjective_rating_shuffled.csv')

    # 1) 먼저 헬퍼 함수들을 최상단에 정의
    def build_from_id_list(id_list, split_name=None):
        rows = []
        for sid in id_list:
            if split_name:
                tex_dir = os.path.join(base_dir, 'split', split_name, 'texture_image')
                nor_dir = os.path.join(base_dir, 'split', split_name, 'normal_map')
                hei_dir = os.path.join(base_dir, 'split', split_name, 'height_map')
            else:
                tex_dir = os.path.join(base_dir, 'original', 'texture_image')
                nor_dir = os.path.join(base_dir, 'original', 'normal_map')
                hei_dir = os.path.join(base_dir, 'original', 'height_map')

            tex = _find_image_path(tex_dir, int(sid), ['.png', '.jpg', '.JPG'])
            nor = _find_image_path(nor_dir, int(sid), ['.png', '.jpg', '.JPG'])
            hei = _find_image_path(hei_dir, int(sid), ['.png', '.jpg', '.JPG'])

            if all([tex, nor, hei]):
                if os.path.exists(csv_path):
                    labels_df = pd.read_csv(csv_path, header=None)
                    idx = int(sid) - 1
                    rough = labels_df.iloc[idx][0]
                else:
                    rough = 0.0

                rows.append({
                    'texture_path': tex,
                    'normal_path': nor,
                    'height_path': hei,
                    'roughness': rough
                })
        return pd.DataFrame(rows)

    # 2) train_ids / valid_ids 있는 경우
    train_ids_path = os.path.join(base_dir, 'split','train_ids.csv')
    valid_ids_path = os.path.join(base_dir, 'split','valid_ids.csv')
    if os.path.exists(train_ids_path) and os.path.exists(valid_ids_path):
        train_ids = pd.read_csv(train_ids_path)['id'].astype(str).tolist()
        valid_ids = pd.read_csv(valid_ids_path)['id'].astype(str).tolist()

        train_df = build_from_id_list(train_ids, split_name='train')
        valid_df = build_from_id_list(valid_ids, split_name='valid')
        test_df = pd.DataFrame([])
        return train_df, valid_df #, test_df

    # otherwise fall back to adjective_rating_shuffled.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    labels_df = pd.read_csv(csv_path, header=None)

    base = Path(base_dir)
    # If data already split into train/valid directories, build per-split DataFrames
    train_dir = base / 'train'
    valid_dir = base / 'valid'
    if train_dir.exists() and valid_dir.exists():
        def build_from_split(split_dir: Path):
            # expect split_dir/texture_image, split_dir/normal_map, split_dir/height_map
            tex = split_dir / 'texture_image'
            nor = split_dir / 'normal_map'
            hei = split_dir / 'height_map'
            if not (tex.exists() and nor.exists() and hei.exists()):
                return pd.DataFrame([])

            tex_ids = {p.stem for p in tex.iterdir() if p.is_file()}
            nor_ids = {p.stem for p in nor.iterdir() if p.is_file()}
            hei_ids = {p.stem for p in hei.iterdir() if p.is_file()}

            ids = sorted(tex_ids & nor_ids & hei_ids, key=lambda x: int(x) if x.isdigit() else x)
            rows = []
            for sid in ids:
                try:
                    idx = int(sid) - 1  # labels CSV is 0-based index matching image number-1
                except Exception:
                    continue
                if idx < 0 or idx >= len(labels_df):
                    continue
                rows.append({
                    'texture_path': str(tex / f"{sid}.png") if (tex / f"{sid}.png").exists() else str(next(tex.glob(f"{sid}.*"))),
                    'normal_path': str(nor / f"{sid}.png") if (nor / f"{sid}.png").exists() else str(next(nor.glob(f"{sid}.*"))),
                    'height_path': str(hei / f"{sid}.png") if (hei / f"{sid}.png").exists() else str(next(hei.glob(f"{sid}.*"))),
                    'roughness': labels_df.iloc[idx][0]
                })
            return pd.DataFrame(rows)

        train_df = build_from_split(train_dir)
        valid_df = build_from_split(valid_dir)
        test_df = pd.DataFrame([])
        return train_df, valid_df, test_df

    # fallback: original behavior (build full list then split)
    exts = ['.png', '.jpg', '.JPG']
    data_rows = []
    for idx in range(len(labels_df)):
        texture_path = _find_image_path(os.path.join(base_dir, "texture_image"), idx+1, exts)
        normal_path = _find_image_path(os.path.join(base_dir, "normal_map"), idx+1, exts)
        height_path = _find_image_path(os.path.join(base_dir, "height_map"), idx+1, exts)
        if all([texture_path, normal_path, height_path]):
            data_rows.append({
                'texture_path': texture_path,
                'normal_path': normal_path,
                'height_path': height_path,
                'roughness': labels_df.iloc[idx][0]
            })

    return split_dataframe(pd.DataFrame(data_rows))


def build_original_dataframe(base_dir="data"):
    csv_path = os.path.join(base_dir, 'original', 'adjective_rating_shuffled.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Original label CSV not found: {csv_path}")

    labels_df = pd.read_csv(csv_path, header=None)
    exts = ['.png', '.jpg', '.JPG']
    data_rows = []
    for idx in range(len(labels_df)):
        texture_path = _find_image_path(os.path.join(base_dir, 'original', 'texture_image'), idx+1, exts)
        normal_path = _find_image_path(os.path.join(base_dir, 'original', 'normal_map'), idx+1, exts)
        height_path = _find_image_path(os.path.join(base_dir, 'original', 'height_map'), idx+1, exts)
        if all([texture_path, normal_path, height_path]):
            data_rows.append({
                'texture_path': texture_path,
                'normal_path': normal_path,
                'height_path': height_path,
                'roughness': labels_df.iloc[idx][0]
            })
    return pd.DataFrame(data_rows)


def split_dataframe(df, valid_size=0.5, test_size=0.15, random_state=42):
    # train_df, test_df = train_test_split(
    #     df, test_size=test_size, random_state=random_state
    # )
    train_df, valid_df = train_test_split(
        df, test_size=valid_size/(1-test_size), random_state=random_state
    )

    return train_df, valid_df#, test_df

def _find_image_path(directory, idx, exts):
    """확장자 순회하며 실제 파일 경로 탐색"""
    for ext in exts:
        path = os.path.join(directory, f"{idx}{ext}")
        if os.path.exists(path):
            return path
    return None