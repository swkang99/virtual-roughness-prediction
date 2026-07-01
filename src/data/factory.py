import numpy as np
from src.data.dataset import FeatureDataset, OriginalDataset, SeparatedDataset
from src.model.feature.feature_extractor import FeatureExtractor

MODEL_DATASET_TYPE = {
    "lr": "separated",
    "svr": "separated",
    "ann": "separated",
    "cnn_1d_scirep": "feature",
    "cnn_1d_4ha": "separated",
    "transformer": "original",
    "cnn_1d_generic": "separated",
    "gated_mlp": "separated",
    "gated_mlp_v2": "separated",
}

def build_feature_base_dataset(full_df, device):
    feature_extractor = FeatureExtractor(device)
    full_features, full_targets = feature_extractor.precompute_features_and_targets(full_df)
    input_dim = full_features.shape[1]
    return FeatureDataset(full_features, full_targets), full_targets, input_dim

def build_original_base_dataset(full_df):
    base_dataset = OriginalDataset(full_df)
    full_targets = full_df["roughness"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return base_dataset, full_targets, None

def build_separated_dataset(full_df, device):
    feature_extractor = FeatureExtractor(device)

    texture_feats, height_feats, normal_feats, all_targets = feature_extractor.precompute_features_and_targets_separated(full_df)

    texture_feats = np.asarray(texture_feats, dtype=np.float32)
    height_feats  = np.asarray(height_feats, dtype=np.float32)
    normal_feats  = np.asarray(normal_feats, dtype=np.float32)
    all_targets   = np.asarray(all_targets, dtype=np.float32).reshape(len(all_targets), -1)

    input_dim = {
        'texture_dim': texture_feats.shape[1],
        'height_dim': height_feats.shape[1],
        'normal_dim': normal_feats.shape[1],
    }

    return (
        SeparatedDataset(texture_feats, height_feats, normal_feats, all_targets),
        all_targets,
        input_dim,
    )

def build_base_dataset(conf, full_df, device):
    dataset_type = MODEL_DATASET_TYPE[conf["model"]]

    if dataset_type == "feature":
        return build_feature_base_dataset(full_df, device)
    elif dataset_type == "original":
        return build_original_base_dataset(full_df)
    elif dataset_type == "separated":
        return build_separated_dataset(full_df, device)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    