import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
    
from src.data.dataframe import build_dataframe_from_file
from src.data.dataset import NormalizedSubset
from src.data.factory import build_base_dataset
from src.model.factory import create_model
from src.model.train import train_one_fold, evaluate_one_fold
from src.utils.metrics import metrics

def loocv(conf, model_builder):
    epochs = int(conf['epochs'])
    batch_size = int(conf['batch_size'])
    lr = float(conf['learning_rate'])
    weight_decay = float(conf['weight_decay'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    full_df = build_dataframe_from_file(conf)

    target_col = (
        "haptic_attribute" if conf['dataset_output'] == 'four_HAs'
        else "roughness"
    )

    base_dataset, full_targets, input_dim = build_base_dataset(conf, full_df, target_col, device)

    predictions = []
    ground_truths = []
    test_image_ids = []

    print(f"\nStarting LOOCV training with {len(full_df)} samples...")
    for test_idx in tqdm(range(len(full_df)), desc="LOOCV", unit="fold"):

        train_indices = [i for i in range(len(full_df)) if i != test_idx]
        test_indices = [test_idx]

        y_train_raw = full_targets[train_indices]   # shape: (N-1, 4) or (N-1, 1)
        y_min = y_train_raw.min(axis=0)
        y_max = y_train_raw.max(axis=0)

        train_dataset = NormalizedSubset(base_dataset, train_indices, y_min, y_max)
        test_dataset  = NormalizedSubset(base_dataset, test_indices, y_min, y_max)

        if input_dim is None:
            model = model_builder(conf, input_dim=None, device=device)
        else: 
            model = model_builder(conf, input_dim=input_dim, device=device)

        model = train_one_fold(
            model=model,
            dataset=train_dataset,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        
        preds, gts = evaluate_one_fold(
            model=model,
            dataset=test_dataset,
            device=device,
            y_min=y_min,
            y_max=y_max,
        )
    
        predictions.append(preds) 
        ground_truths.append(gts)
        test_image_ids.append(test_idx)

    predictions = np.array(predictions, dtype=np.float32)
    predictions = predictions.reshape(predictions.shape[0], -1)
    ground_truths = np.array(ground_truths, dtype=np.float32)
    ground_truths = ground_truths.reshape(ground_truths.shape[0], -1)

    mae_per_output = mean_absolute_error(ground_truths, predictions, multioutput='raw_values')
    rmse_per_output = np.sqrt(np.mean((ground_truths - predictions) ** 2, axis=0))

    metrics(
        conf, 
        mae_per_output=mae_per_output, 
        rmse_per_output=rmse_per_output,
        predictions=predictions,
        ground_truths=ground_truths,
        test_image_ids=test_image_ids,
    )

def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    loocv(conf, create_model)

if __name__ == '__main__':
    main()