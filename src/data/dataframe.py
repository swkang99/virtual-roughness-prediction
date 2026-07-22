import os
import pandas as pd
from tqdm import tqdm
from src.data.texture_maps import process_texture
    
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
        height_dir, normal_dir = process_texture(tex_path, save_texture_maps=False)

        roughness_values = [v[1] for v in label_map.values() if int(v[0]) == int(sid)]

        row = {
            'texture_path': str(tex_path),
            'normal_path': normal_dir,
            'height_path': height_dir,
            'roughness': float(roughness_values[0]),
        }
        rows.append(row)

    return pd.DataFrame(rows)
