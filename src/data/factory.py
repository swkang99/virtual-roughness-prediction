import numpy as np
from src.data.dataset import FeatureDataset, OriginalDataset
from src.model.feature.cnn_1d_4ha import FeatureExtractor

MODEL_DATASET_TYPE = {
    "lr": "feature",
    "svr": "feature",
    "ann": "feature",
    "cnn_1d_scirep": "feature",
    "cnn_1d_4ha": "feature",
    "transformer": "original",
}

def build_feature_base_dataset(conf, full_df, target_col, device):
    feature_extractor = FeatureExtractor(device)
    full_features, full_targets = feature_extractor.precompute_features_and_targets(
        full_df, conf, target_col
    )
    return FeatureDataset(full_features, full_targets), full_targets

def build_original_base_dataset(conf, full_df, target_col, device):
    base_dataset = OriginalDataset(full_df, conf, target_col)

    if conf["dataset_output"] == "four_HAs":
        full_targets = np.stack(full_df[target_col].to_list()).astype(np.float32)
    else:
        full_targets = full_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    return base_dataset, full_targets

def build_base_dataset(conf, full_df, target_col, device):
    dataset_type = MODEL_DATASET_TYPE[conf["model"]]

    if dataset_type == "feature":
        return build_feature_base_dataset(conf, full_df, target_col, device)
    elif dataset_type == "original":
        return build_original_base_dataset(conf, full_df, target_col, device)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")