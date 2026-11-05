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


def build_tier_dataframe(tier_dir, label_csv_path, tier_name=None):
    """Build a (texture_path, height_path, normal_path, roughness, texture, tier) dataframe
    from one Tier's `Patches/{Texture,Height,Normal}/` folder, as written by
    `python -m src.data.prepare_tier_patches` (see that script's docstring for the exact
    patch filename convention: `<stem>_patch_XXXX(.png|_height_map_gray.png|_normal_map_rgb.png)`,
    where `<stem>` starts with the texture id, e.g. "1_Lined Wood1" or "A1_Silk").

    `label_csv_path` is that Tier's own label file (columns: texture, name, label, quintile;
    `label` is already on the unified 0-100 APT scale - see claude/whc2027_apt_relabeling_strategy.md
    §3.1/§4. The `texture` column is matched as a string, since Tier A's ids ("A1".."A8") are not
    numeric while Tier B / Tier A' ids are plain integers written as strings ("1", "10", ...).
    """
    tier_dir = Path(tier_dir)
    labels = pd.read_csv(label_csv_path, dtype={'texture': str})
    label_map = dict(zip(labels['texture'], labels['label']))

    tex_dir = tier_dir / 'Patches' / 'Texture'
    height_dir = tier_dir / 'Patches' / 'Height'
    normal_dir = tier_dir / 'Patches' / 'Normal'
    if not tex_dir.is_dir():
        raise FileNotFoundError(
            f'{tex_dir} missing - run `python -m src.data.prepare_tier_patches --tier "{tier_dir.name}"` first'
        )

    patch_files = sorted(p for p in tex_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})

    rows = []
    for tex_path in tqdm(patch_files, total=len(patch_files), desc=f'Build Dataframe from {tier_dir.name}'):
        stem = tex_path.stem
        if '_patch_' not in stem:
            raise ValueError(f'{tex_path}: expected a "..._patch_XXXX" filename')
        base = stem.rsplit('_patch_', 1)[0]
        texture_id = base.split('_')[0]
        if texture_id not in label_map:
            raise KeyError(f'{tex_path}: texture id {texture_id!r} not found in {label_csv_path}')

        h_path = height_dir / f'{stem}_height_map_gray.png'
        n_path = normal_dir / f'{stem}_normal_map_rgb.png'
        if not (h_path.exists() and n_path.exists()):
            raise FileNotFoundError(f'{h_path} / {n_path} missing for {tex_path}')

        rows.append({
            'texture_path': str(tex_path),
            'height_path': str(h_path),
            'normal_path': str(n_path),
            'roughness': float(label_map[texture_id]),
            'texture': texture_id,
            'tier': tier_name if tier_name is not None else tier_dir.name,
        })

    return pd.DataFrame(rows)
