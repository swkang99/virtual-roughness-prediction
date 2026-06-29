import os
from pathlib import Path
from numbers import Number

import pandas as pd

from src.data.texture_maps import process_texture


def _to_sid(value):
    if pd.isna(value):
        return None

    if isinstance(value, Number):
        if float(value).is_integer():
            return str(int(value))

    return str(value)


def _apply_normalization(values, normalized):
    result = []

    for v in values:
        if isinstance(v, Number) and normalized:
            result.append(v + 50)
        else:
            result.append(v)

    return result


def _load_ha_labels(csv_path, header=None, normalized=True):
    if not os.path.exists(csv_path):
        return {}

    labels_df = pd.read_csv(csv_path, header=header)

    # split csv 형식:
    # file_name,roughness
    # 3,40.1827
    # 4,35.581
    if header is not None and "file_name" in labels_df.columns:
        value_cols = [
            col for col in labels_df.columns
            if col != "file_name"
        ]

        label_map = {}

        for _, row in labels_df.iterrows():
            sid = _to_sid(row["file_name"])
            values = [
                row[col]
                for col in value_cols
            ]

            label_map[sid] = _apply_normalization(values, normalized)

        return label_map

    # 기존 original csv 형식:
    # id column 없이 row 순서가 1, 2, 3, ... 에 대응
    label_map = {
        str(i + 1): _apply_normalization(
            labels_df.iloc[i].tolist(),
            normalized,
        )
        for i in range(len(labels_df))
    }

    return label_map


def _extract_label_id_from_texture_name(texture_stem):
    if "_patch_" in texture_stem:
        return texture_stem.split("_patch_", 1)[0]

    return texture_stem


def _texture_sort_key(path):
    stem = path.stem
    label_id = _extract_label_id_from_texture_name(stem)

    if label_id.isdigit():
        label_key = (0, int(label_id))
    else:
        label_key = (1, label_id)

    if "_patch_" in stem:
        patch_id = stem.split("_patch_", 1)[1]

        if patch_id.isdigit():
            patch_key = (0, int(patch_id))
        else:
            patch_key = (1, patch_id)
    else:
        patch_key = (0, -1)

    return label_key, patch_key


def build_dataframe_from_file(conf, texture_path, label_path, header):
    texture_path = Path(texture_path)
    label_path = Path(label_path)

    label_map = _load_ha_labels(
        label_path,
        header=header,
        normalized=False,
    )

    texture_files = [
        p for p in texture_path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    texture_files = sorted(texture_files, key=_texture_sort_key)

    rows = []

    for tex_path in texture_files:
        label_id = _extract_label_id_from_texture_name(tex_path.stem)

        haptic_attribute_list = label_map.get(label_id)

        if haptic_attribute_list is None:
            raise ValueError(
                f"No label found for texture file: {tex_path.name}. "
                f"Expected label key: {label_id}"
            )

        row = {
            "texture_path": str(tex_path),
            "source_id": label_id,
            "patch_id": tex_path.stem,
        }

        if conf["dataset_input"] == "texture_maps":
            height_path, normal_path = process_texture(
                tex_path,
                save_texture_maps=True,
            )

            row.update({
                "height_path": str(height_path),
                "normal_path": str(normal_path),
            })

        elif conf["dataset_input"] == "texture_image":
            pass

        else:
            raise ValueError(f"Unsupported dataset_input: {conf['dataset_input']}")

        if conf["dataset_output"] == "four_HAs":
            row["haptic_attribute"] = haptic_attribute_list

        elif conf["dataset_output"] == "roughness":
            row["roughness"] = float(haptic_attribute_list[0])

        else:
            raise ValueError(f"Unsupported dataset_output: {conf['dataset_output']}")

        rows.append(row)

    return pd.DataFrame(rows)


def _find_label_file(split_path, candidates):
    split_path = Path(split_path)

    for name in candidates:
        path = split_path / name

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No label csv found in {split_path}. "
        f"Checked: {candidates}"
    )


def build_split_dataframes(conf):
    train_path = Path(conf["data_train_path"])
    val_path = Path(conf["data_val_path"])
    test_path = Path(conf["data_test_path"])

    train_df = build_dataframe_from_file(
        conf,
        texture_path=train_path / Path(conf["data_patch_path"]),
        label_path=_find_label_file(train_path, ["train.csv"]),
        header=0,
    )

    val_df = build_dataframe_from_file(
        conf,
        texture_path=val_path / Path(conf["data_patch_path"]),
        label_path=_find_label_file(val_path, ["valid.csv", "val.csv"]),
        header=0,
    )

    test_df = build_dataframe_from_file(
        conf,
        texture_path=test_path / Path(conf["data_patch_path"]),
        label_path=_find_label_file(test_path, ["test.csv"]),
        header=0,
    )

    return train_df, val_df, test_df