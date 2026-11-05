"""
Shared batch-handling utilities for the per-model input conventions (texture/height/normal image
triplet vs. a flat feature vector, and the gated_mlp / gated_mlp_v2 models' shared-feature inputs).

Used by test/cv/cv_experiment.py and test/holdout/holdout_eval.py (the active Tier-based CV / holdout
pipelines). The old Trainer class and its
train/val/test split-path pipeline (which this file used to also contain) were removed: the data
under data/train, data/val, data/test no longer exists (replaced by the Tier-based pool under
data/trustworthy - see config.yaml's data_tier_root), so that pipeline could no longer run.
test/experiment.py and train_and_test_all_model_repeatedly.sh, which only existed to drive that
pipeline, are dead for the same reason.
"""
import torch


def is_gated_mlp(model):
    module_name = model.__class__.__module__
    return module_name.startswith('src.model.prediction.proposed.gated_mlp')


def is_end_to_end(model):
    module_name = model.__class__.__module__
    is_transformer = module_name.startswith('src.model.prediction.proposed.transformer')
    is_cnn_1d_simple = module_name.startswith('src.model.prediction.compared.cnn_1d.simple')
    return is_transformer or is_cnn_1d_simple


def is_torch_model(model):
    return isinstance(model, torch.nn.Module)


def prepare_batch_by_model(batch, model, device):
    """
    PatchDataset :
        batch = (texture, height, normal, y)

    PatchFeatureDataset :
        batch = (x, y)

    gated_mlp / gated_mlp_v2:
        Use a single feature x as a common input for the three inputs: texture_feat, height_feat, and normal_feat
    """

    if is_gated_mlp(model):
        if len(batch) == 2:
            x, y = batch
            x = x.to(device).float()
            y = y.to(device).float()
            return (x, x, x), y

        if len(batch) == 4:
            texture, height, normal, y = batch
            texture = texture.to(device).float()
            height = height.to(device).float()
            normal = normal.to(device).float()
            y = y.to(device).float()
            return (texture, height, normal), y

        raise ValueError(f"Unsupported batch format for gated_mlp with length {len(batch)}")

    if is_end_to_end(model):
        texture, height, normal, y = batch
        texture = texture.to(device).float()
        height = height.to(device).float()
        normal = normal.to(device).float()
        y = y.to(device).float()
        return (texture, height, normal), y

    if len(batch) == 4:
        texture, height, normal, y = batch
        x = torch.cat([texture, height, normal], dim=1).to(device).float()
        y = y.to(device).float()
        return x, y

    if len(batch) == 2:
        x, y = batch
        x = x.to(device).float()
        y = y.to(device).float()
        return x, y

    raise ValueError(f"Unsupported batch format with length {len(batch)}")


def forward_by_model(model, inputs):
    if is_gated_mlp(model) or is_end_to_end(model):
        texture, height, normal = inputs
        return model(texture, height, normal)
    return model(inputs)
