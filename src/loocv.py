import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
    
from src.data.dataframe import build_dataframe_from_file
from src.data.dataset import load_cnn_1d_dataset, load_original_dataset
from src.model.train import train_one_fold, evaluate_one_fold
from src.model.prediction.compared.cnn_1d_4ha import CNN1D4HA
from src.utils.metrics import metrics

def loocv(conf):
    epochs = int(conf['epochs'])
    batch_size = int(conf['batch_size'])
    lr = float(conf['learning_rate'])
    weight_decay = float(conf['weight_decay'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    full_df = build_dataframe_from_file()

    predictions = []
    ground_truths = []
    test_image_ids = []

    print(f"\nStarting LOOCV training with {len(full_df)} samples...")

    for test_idx in tqdm(range(len(full_df)), desc="LOOCV", unit="fold"):

        train_mask = np.ones(len(full_df), dtype=bool)
        train_mask[test_idx] = False

        train_df = full_df[train_mask].reset_index(drop=True)
        test_df  = full_df.iloc[test_idx:test_idx+1].reset_index(drop=True)

        if conf['dataset_output'] == 'four_HAs':
            target_col = "haptic_attribute"
            y_train_list = full_df.loc[train_mask, target_col].tolist()
            y_train_raw = np.array(y_train_list, dtype=np.float32) # (N, 4)
        elif conf['dataset_output'] == 'roughness':
            target_col = "roughness"
            y_train_raw = full_df.loc[train_mask, target_col].to_numpy(dtype=np.float32)
            y_train_raw = y_train_raw.reshape(-1, 1) # (N, 1)

        y_min = y_train_raw.min(axis=0)  # (4,) or (1,)
        y_max = y_train_raw.max(axis=0)  # (4,) or (1,)

        model = CNN1D4HA(conf).to(device)

        if conf['model'] == 'cnn_1d_4ha':
            train_dataset = load_cnn_1d_dataset(train_df, conf, target_col, y_min=y_min, y_max=y_max)
            test_dataset  = load_cnn_1d_dataset(test_df, conf, target_col, y_min=y_min, y_max=y_max)
        else:
            train_dataset = load_original_dataset(train_df)
            test_dataset = load_original_dataset(test_df)
            

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
        test_image_ids.append(test_df.loc[0, "texture_path"])

    predictions = np.array(predictions, dtype=np.float32)
    ground_truths = np.array(ground_truths, dtype=np.float32)

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

    loocv(conf)

if __name__ == '__main__':
    main()