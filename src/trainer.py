"""
Trainer class for encapsulating model training and evaluation logic.
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from src.data.dataframe import build_dataframe_from_file
from src.data.dataset import NormalizedSubset, dataset_to_numpy
from src.data.factory import build_base_dataset, MODEL_DATASET_TYPE
from src.model.prediction.proposed.gated_mlp import GatedFusionRegressor
from src.model.prediction.proposed.gated_mlp_v2 import GatedFusionRegressorV2
from src.utils.metrics import metrics


def is_gated_mlp(model):
    return isinstance(model, GatedFusionRegressor) or isinstance(model, GatedFusionRegressorV2)


def is_torch_model(model):
    return isinstance(model, torch.nn.Module)


def prepare_batch_by_model(batch, model, device):
    """
    SeparatedDataset 기준:
        batch = (texture, height, normal, y)

    FeatureDataset 기준:
        batch = (x, y)
    """
    if is_gated_mlp(model):
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
    if is_gated_mlp(model):
        texture, height, normal = inputs
        return model(texture, height, normal)
    return model(inputs)


def train_one_fold(model, dataset, device, epochs, batch_size, lr, weight_decay):
    if is_torch_model(model):
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        model.to(device)
        model.train()

        for _ in tqdm(range(epochs), desc="Train One Fold"):
            for batch in train_loader:
                inputs, y = prepare_batch_by_model(batch, model, device)

                optimizer.zero_grad()
                pred = forward_by_model(model, inputs)

                if pred.dim() > y.dim():
                    pred = pred.squeeze(-1)

                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

        return model

    X_train, y_train = dataset_to_numpy(dataset)
    model.fit(X_train, y_train)
    return model


def evaluate_one_fold(model, dataset, device, y_min, y_max, batch_size=32):
    if is_torch_model(model):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        model.to(device)
        model.eval()

        pred_norm_list = []
        y_norm_list = []

        with torch.no_grad():
            for batch in loader:
                inputs, y = prepare_batch_by_model(batch, model, device)

                pred = forward_by_model(model, inputs)

                if pred.dim() > y.dim():
                    pred = pred.squeeze(-1)

                pred_norm_list.append(pred.detach().cpu().numpy())
                y_norm_list.append(y.detach().cpu().numpy())

        pred_norm = np.concatenate(pred_norm_list, axis=0)
        y_norm = np.concatenate(y_norm_list, axis=0)

    else:
        X_test, y_norm = dataset_to_numpy(dataset)
        pred_norm = model.predict(X_test)

        if pred_norm.ndim == 1:
            pred_norm = pred_norm.reshape(-1, 1)
        if y_norm.ndim == 1:
            y_norm = y_norm.reshape(-1, 1)

    y_min = y_min.item()
    y_max = y_max.item()
    pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
    gt_raw = y_norm * (y_max - y_min + 1e-8) + y_min

    return pred_raw, gt_raw


class Trainer:
    """
    Unified trainer for managing model training and evaluation workflows.
    
    Supports:
    - LOOCV (Leave-One-Out Cross-Validation)
    - Train/Val/Test split-based training
    - Model checkpointing
    - Metric computation
    """

    _split_cache = {}
    
    def __init__(self, conf, model_builder, device=None, work_dir=None):
        """
        Initialize the Trainer.
        
        Args:
            conf (dict): Configuration dictionary from config.yaml
            model_builder (callable): Function to build model instances
                signature: model_builder(conf, input_dim=None, device=None) -> model
            device (torch.device, optional): Device to use. Defaults to CUDA if available, else CPU.
            work_dir (str or Path, optional): Directory for saving checkpoints/results.
                Defaults to conf['trainer']['save_dir'] or current directory.
        """
        self.conf = conf
        self.model_builder = model_builder
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        
        trainer_conf = conf.get('trainer', {})
        save_dir = work_dir or trainer_conf.get('save_dir', '.')
        self.work_dir = Path(save_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract hyperparameters from config
        self.epochs = int(conf.get('epochs', 100))
        self.batch_size = int(conf.get('batch_size', 32))
        self.lr = float(conf.get('learning_rate', 1e-3))
        self.weight_decay = float(conf.get('weight_decay', 1e-5))
        self.seed = conf.get('seed', 42)
        self.verbose = conf.get('verbose', True)

        self.input_dim = None
    
    def set_seed(self, seed=None):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed (int, optional): Seed value. If None, uses self.seed.
        """
        if seed is None:
            seed = self.seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_from_dataframe(self, full_df):
        """
        Prepare base dataset and targets from a dataframe.
        
        Args:
            full_df (pd.DataFrame): Full dataframe with all samples.
        
        Returns:
            tuple: (base_dataset, full_targets, input_dim)
                - base_dataset: Dataset object
                - full_targets: NumPy array of shape (N, D) or (N,)
                - input_dim: Input dimension or None
        """
        target_col = "roughness"
        
        base_dataset, full_targets, input_dim = build_base_dataset(
            self.conf, full_df, target_col, self.device
        )
        return base_dataset, full_targets, input_dim
    
    def compute_y_norm(self, y_train_raw):
        """
        Compute normalization bounds from training targets.
        
        Args:
            y_train_raw (np.ndarray): Training targets, shape (M, D) or (M,)
        
        Returns:
            tuple: (y_min, y_max) normalized bounds
        """
        if y_train_raw.ndim == 1:
            y_train_raw = y_train_raw.reshape(-1, 1)
        
        y_min = y_train_raw.min(axis=0)
        y_max = y_train_raw.max(axis=0)
        return y_min, y_max
    
    def build_split_dataframe(self, split_name):
        """
        Build a dataframe for a named split using config paths.
        """
        base_path = Path(self.conf[f"data_{split_name}_path"])
        texture_path = base_path / Path(self.conf['data_patch_path'])
        label_path = base_path / Path(f"{split_name}.csv")
        return build_dataframe_from_file(
            texture_path=texture_path,
            label_path=label_path,
            header=0,
        )
    
    def make_normalized_subset(self, base_dataset, indices, y_min, y_max):
        """
        Create a normalized dataset subset.
        
        Args:
            base_dataset: Base dataset object
            indices (list): Indices to include
            y_min (np.ndarray): Min normalization bound
            y_max (np.ndarray): Max normalization bound
        
        Returns:
            NormalizedSubset: Normalized dataset
        """
        return NormalizedSubset(base_dataset, indices, y_min, y_max)
    
    def _normalize_input_dim(self, input_dim):
        """
        Normalize input_dim to the format expected by the selected model.
        """
        model_name = self.conf.get('model')

        if input_dim is None:
            return None

        if model_name in {'ann', 'lr', 'svr', 'cnn_1d_scirep', 'cnn_1d_4ha', 'cnn_1d_generic'}:
            if isinstance(input_dim, dict):
                if 'texture_dim' in input_dim:
                    return input_dim['texture_dim']
                if 'input_dim' in input_dim:
                    return input_dim['input_dim']
                return None
            return input_dim

        if model_name in {'gated_mlp', 'gated_mlp_v2'}:
            if isinstance(input_dim, dict):
                return input_dim
            return {'texture_dim': input_dim, 'height_dim': input_dim, 'normal_dim': input_dim}

        return input_dim

    def build_model(self, input_dim=None):
        """
        Build a model using the provided model_builder.
        
        Args:
            input_dim (int, optional): Input dimension. If None, model_builder handles it.
        
        Returns:
            model: Instantiated model
        """
        normalized_input_dim = self._normalize_input_dim(input_dim if input_dim is not None else self.input_dim)
        return self.model_builder(self.conf, input_dim=normalized_input_dim, device=self.device)
    
    def fit(self, train_dataset, val_dataset=None):
        """
        Train a model on train_dataset with optional validation.
        
        Args:
            train_dataset (NormalizedSubset): Training dataset
            val_dataset (NormalizedSubset, optional): Validation dataset
        
        Returns:
            tuple: (trained_model, val_results)
                - trained_model: Trained model
                - val_results: Dict with 'preds', 'gts' if val_dataset provided, else None
        """
        # Build model
        model = self.build_model(input_dim=self.input_dim)
        
        print(f"Fitting model : {type(model).__name__}")

        # Train
        model = train_one_fold(
            model=model,
            dataset=train_dataset,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Validate (if provided)
        val_results = None
        if val_dataset is not None:
            y_min = val_dataset.y_min
            y_max = val_dataset.y_max
            preds, gts = evaluate_one_fold(
                model=model,
                dataset=val_dataset,
                device=self.device,
                y_min=y_min,
                y_max=y_max,
            )
            val_results = {'preds': preds, 'gts': gts}
        
        return model, val_results
    
    def evaluate(self, model, dataset):
        """
        Evaluate model on a dataset.
        
        Args:
            model: Trained model
            dataset (NormalizedSubset): Dataset to evaluate on
        
        Returns:
            tuple: (predictions, ground_truths)
        """
        y_min = dataset.y_min
        y_max = dataset.y_max
        preds, gts = evaluate_one_fold(
            model=model,
            dataset=dataset,
            device=self.device,
            y_min=y_min,
            y_max=y_max,
        )
        return preds, gts
    
    def _get_or_build_split_dataset(self, split_name, y_min=None, y_max=None):
        """
        Build and cache a normalized split dataset.
        The cache is shared across Trainer instances so repeated experiments
        can reuse the same split preprocessing work.
        """
        dataset_type = MODEL_DATASET_TYPE.get(self.conf.get('model'), 'unknown')
        cache_key = (
            split_name,
            dataset_type,
            self.conf.get('data_patch_path'),
            self.conf.get(f'data_{split_name}_path'),
        )

        if cache_key in self._split_cache:
            if self.verbose:
                print(f"Using cached split dataset for {split_name}")
            return self._split_cache[cache_key]

        if split_name in {'train', 'val', 'test'}:
            df = self.build_split_dataframe(split_name)
        else:
            raise ValueError(f'Unsupported split name: {split_name}')

        base_dataset, targets, input_dim = build_base_dataset(
            self.conf, df, self.device
        )

        if y_min is None or y_max is None:
            y_min, y_max = self.compute_y_norm(targets)

        dataset = self.make_normalized_subset(
            base_dataset, list(range(len(base_dataset))), y_min, y_max
        )

        cache_entry = {
            'dataset': dataset,
            'input_dim': input_dim,
            'y_min': y_min,
            'y_max': y_max,
        }
        self._split_cache[cache_key] = cache_entry
        return cache_entry

    def fit_splits(self, train_df=None, val_df=None, test_df=None, compute_metrics=True):
        """
        Train/validate/test using separate dataframes for each split.
        
        Args:
            train_df (pd.DataFrame, optional): Training dataframe.
                If None, built from config paths `data_train_path` and `data_patch_path`.
            val_df (pd.DataFrame, optional): Validation dataframe.
                If None, built from config paths `data_val_path` and `data_patch_path`.
            test_df (pd.DataFrame, optional): Test dataframe.
                If None, built from config paths `data_test_path` and `data_patch_path`.
            compute_metrics (bool): Whether to compute and log metrics.
        
        Returns:
            dict: Results dictionary with keys:
                - 'model': Trained model
                - 'val': Validation results (dict with 'preds', 'gts') or None
                - 'test': Test results (dict with 'preds', 'gts') or None
                - 'metrics': Aggregated metrics dict or None
        """

        if train_df is not None:
            train_base, train_targets, input_dim = build_base_dataset(
                self.conf, train_df, self.device
            )
            y_min, y_max = self.compute_y_norm(train_targets)
            train_dataset = self.make_normalized_subset(
                train_base, list(range(len(train_base))), y_min, y_max
            )
            self.input_dim = input_dim
        else:
            train_cache = self._get_or_build_split_dataset('train')
            train_dataset = train_cache['dataset']
            self.input_dim = train_cache['input_dim']

        if val_df is not None:
            val_base, _, _ = build_base_dataset(self.conf, val_df, self.device)
            val_dataset = self.make_normalized_subset(
                val_base, list(range(len(val_base))), train_dataset.y_min, train_dataset.y_max
            )
        else:
            val_cache = self._get_or_build_split_dataset(
                'val', train_dataset.y_min, train_dataset.y_max
            )
            val_dataset = val_cache['dataset']

        if test_df is not None:
            test_base, _, _ = build_base_dataset(self.conf, test_df, self.device)
            test_dataset = self.make_normalized_subset(
                test_base, list(range(len(test_base))), train_dataset.y_min, train_dataset.y_max
            )
        else:
            test_cache = self._get_or_build_split_dataset(
                'test', train_dataset.y_min, train_dataset.y_max
            )
            test_dataset = test_cache['dataset']

        model, val_results = self.fit(train_dataset, val_dataset=val_dataset)
        preds, gts = self.evaluate(model, test_dataset)
        test_results = {'preds': preds, 'gts': gts}

        # Optionally compute metrics
        metrics_dict = None
        if compute_metrics:
            metrics_dict = {}
            
            if val_results is not None:
                val_preds = val_results['preds']
                val_gts = val_results['gts']
                val_mae = mean_absolute_error(val_gts, val_preds, multioutput='raw_values')
                val_rmse = np.sqrt(np.mean((val_gts - val_preds) ** 2, axis=0))
                metrics_dict['val'] = {'mae': val_mae, 'rmse': val_rmse}
            
            if test_results is not None:
                test_preds = test_results['preds']
                test_gts = test_results['gts']
                test_mae = mean_absolute_error(test_gts, test_preds, multioutput='raw_values')
                test_rmse = np.sqrt(np.mean((test_gts - test_preds) ** 2, axis=0))
                metrics_dict['test'] = {'mae': test_mae, 'rmse': test_rmse}

        return {
            'model': model,
            'val': val_results,
            'test': test_results,
            'metrics': metrics_dict,
        }
    
    def save_checkpoint(self, model, fname):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            fname (str or Path): Checkpoint filename
        """
        fpath = self.work_dir / fname if not Path(fname).is_absolute() else Path(fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), fpath)
        else:
            # For non-torch models (sklearn, etc.)
            import pickle
            with open(fpath, 'wb') as f:
                pickle.dump(model, f)
        
        if self.verbose:
            print(f"Checkpoint saved to {fpath}")
    
    def load_checkpoint(self, fname):
        """
        Load model checkpoint.
        Note: Creates a new model instance and loads weights into it.
        
        Args:
            fname (str or Path): Checkpoint filename
        
        Returns:
            model: Loaded model
        """
        fpath = self.work_dir / fname if not Path(fname).is_absolute() else Path(fname)
        
        # Build new model
        model = self.build_model()
        
        if isinstance(model, torch.nn.Module):
            state_dict = torch.load(fpath, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # For non-torch models
            import pickle
            with open(fpath, 'rb') as f:
                loaded_model = pickle.load(f)
            return loaded_model
        
        if self.verbose:
            print(f"Checkpoint loaded from {fpath}")
        
        return model
