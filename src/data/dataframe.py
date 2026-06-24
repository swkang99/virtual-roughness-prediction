import os
import pandas as pd
from src.data.texture_maps import process_texture
    
def _load_ha_labels(csv_path, header=None, normalized=True):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=header)
    ha_list = {
        str(i + 1): [
            v + 50 if isinstance(v, (int, float)) and normalized else v
            for v in labels_df.iloc[i].tolist()
        ]
        for i in range(len(labels_df))
    }
    return ha_list


def build_dataframe_from_file(conf, texture_path, label_path, header):

    label_map = _load_ha_labels(label_path, header=header, normalized=False)

    texture_files = [
        p for p in texture_path.iterdir()
        if p.suffix.lower() in {'.png', '.jpg'}
    ]
    texture_files = sorted(texture_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for tex_path in texture_files:
        sid = tex_path.stem
        height_dir, normal_dir = process_texture(tex_path, save_texture_maps=True)
        
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

# train_df = build_dataframe_from_file(
#     conf, 
#     texture_path = Path(conf['data_train_path']) / Path(conf['data_patch_path']),
#     label_path = Path(conf['data_train_path']) / Path('train.csv'),
#     header=0,
# )

# val_df = build_dataframe_from_file(
#     conf, 
#     texture_path = Path(conf['data_val_path']) / Path(conf['data_patch_path']),
#     label_path = Path(conf['data_val_path']) / Path('val.csv'),
#     header=0,
# )

# test_df = build_dataframe_from_file(
#     conf, 
#     texture_path = Path(conf['data_test_path']) / Path(conf['data_patch_path']),
#     label_path = Path(conf['data_test_path']) / Path('test.csv'),
#     header=0,
# )