import pandas as pd
import os
from pathlib import Path
from src.data.texture_maps import process_texture
import yaml

conf = yaml.safe_load('config.yaml')
    
def _load_roughness_labels(csv_path):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=None)
    ha_list = {str(i + 1): [v + 50 if isinstance(v, (int, float)) else v for v in labels_df.iloc[i].tolist()] for i in range(len(labels_df))}
    return ha_list

def build_dataframe_from_file():
    base_path = conf['data_base_path']
    image_path = conf['data_image_path']
    label_path = conf['data_label_path']

    texture_path =Path(base_path / image_path)
    csv_path = Path(base_path / label_path)

    label_map = _load_roughness_labels(csv_path)

    texture_files = [
        p for p in texture_path.iterdir()
        if p.suffix.lower() in {'.png', '.jpg'}
    ]
    texture_files = sorted(texture_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for tex_path in texture_files:
        sid = tex_path.stem
        height_dir, normal_dir = process_texture(tex_path)
        
        haptic_attribute_list = label_map.get(sid)

        row = {
            'texture_path': str(tex_path),
        }

        if conf['dataset_input'] == 'texture_maps':
            row.update({
                'normal_path': normal_dir,
                'height_path': height_dir,
            })
        elif conf['dataset_input'] != 'texture_image':
            raise ValueError(f"Unsupported dataset_input: {conf['dataset_input']}")

        if conf['dataset_output'] == 'four_HAs':
            row['haptic_attribute'] = haptic_attribute_list
        elif conf['dataset_output'] == 'roughness':
            row['roughness'] = float(haptic_attribute_list[0])
        else:
            raise ValueError(f"Unsupported dataset_output: {conf['dataset_output']}")

        rows.append(row)

    return pd.DataFrame(rows)