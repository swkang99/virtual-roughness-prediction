import numpy as np
from src.data.dataset import PatchDataset, PatchFeatureDataset
from src.model.feature.feature_extractor import FeatureExtractor

MODEL_DATASET_TYPE = {
    "lr": "feature",
    "svr": "feature",
    "ann": "feature",
    "cnn_1d_4ha": "feature",
    "cnn_1d_generic": "maps",
    "transformer": "maps",
    "gated_mlp": "feature",
    "gated_mlp_v2": "feature",
}

def build_feature_base_dataset(full_df, device):
    feature_extractor = FeatureExtractor(device)
    full_features, full_targets = feature_extractor.precompute_features_and_targets(full_df)
    input_dim = full_features.shape[1]
    return PatchFeatureDataset(full_features, full_targets), full_targets, input_dim

def build_maps_base_dataset(full_df):
    base_dataset = PatchDataset(full_df)
    full_targets = full_df["roughness"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return base_dataset, full_targets, None

def build_base_dataset(conf, full_df, device):
    dataset_type = MODEL_DATASET_TYPE[conf["model"]]

    if dataset_type == "feature":
        return build_feature_base_dataset(full_df, device)
    elif dataset_type == "maps":
        return build_maps_base_dataset(full_df)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    