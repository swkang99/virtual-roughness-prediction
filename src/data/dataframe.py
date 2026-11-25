from pathlib import Path

import pandas as pd
from tqdm import tqdm


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


def build_tier_a_dataframe(conf, data_root=None):
    """Tier A (8 native APT textures, string ids 'A1'..'A8') - never part of the Tier B / Tier A' train
    pool (see build_tier_dataframe() above / claude/whc2027_apt_relabeling_strategy.md §4). Unlike
    Tier B / Tier A', ids are NOT cast to int.

    Shared by two consumers (see §18.1 of the strategy doc):
      - test/holdout/holdout_eval.py (role 2): the single full-pool model's one-shot Tier A evaluation.
      - test/cv/cv_experiment.py's --eval_tier_a (role 1): each CV fold model's Tier A predictions,
        reused (never retrained) as the repeated-evaluation sample for the confirmatory statistical
        tests, once the recipe is 100% locked. This function itself does not care which caller is
        using it; it is identical "load Tier A" logic either way, moved here (2026-11-25) so both
        callers share one implementation instead of cv_experiment.py reimplementing holdout_eval.py's.
    """
    tier_root = Path(data_root) if data_root else Path(conf.get('data_tier_root', 'data/trustworthy'))
    tier_dir = tier_root / 'Tier A'
    df = build_tier_dataframe(tier_dir, tier_dir / 'Tier A.csv', tier_name='A')
    df['image_id'] = df['texture_path'].map(lambda p: Path(p).stem)
    df['patch_idx'] = df['image_id'].map(lambda s: int(s.split('_')[-1]))
    return df.reset_index(drop=True)
