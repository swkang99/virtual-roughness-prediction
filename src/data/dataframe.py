import os
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Height / normal map patches: crops of the Photoshop-generated maps at the texture-patch positions,
# prepared once by `python -m src.data.crop_photoshop_maps` (data/<split>/<map_dir>/).
with open('config.yaml', 'r', encoding='utf-8') as _f:
    MAP_DIR = (yaml.safe_load(_f) or {}).get('map_dir', 'ps_maps')


def _load_ha_labels(csv_path, header=None):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=header)
    ha_list = {
        str(i + 1): [
            v for v in labels_df.iloc[i].tolist()
        ]
        for i in range(len(labels_df))
    }
    return ha_list


def _map_paths(tex_path):
    d = Path(tex_path).parent.parent / MAP_DIR
    h, n = d / f'{tex_path.stem}_height_map_gray.png', d / f'{tex_path.stem}_normal_map_rgb.png'
    if not (h.exists() and n.exists()):
        raise FileNotFoundError(f'{h} / {n} missing - run `python -m src.data.crop_photoshop_maps` first')
    return str(h), str(n)


def build_dataframe_from_file(texture_path, label_path, header):

    label_map = _load_ha_labels(label_path, header=header)

    texture_files = [
        p for p in texture_path.iterdir()
        if p.suffix.lower() in {'.png', '.jpg'}
    ]
    texture_files = sorted(texture_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for tex_path in tqdm(texture_files, total=len(texture_files), desc="Build Dataframe from file"):
        sid = tex_path.stem.split('_')[0]
        height_dir, normal_dir = _map_paths(tex_path)

        roughness_values = [v[1] for v in label_map.values() if int(v[0]) == int(sid)]

        row = {
            'texture_path': str(tex_path),
            'normal_path': normal_dir,
            'height_path': height_dir,
            'roughness': float(roughness_values[0]),
        }
        rows.append(row)

    return pd.DataFrame(rows)
