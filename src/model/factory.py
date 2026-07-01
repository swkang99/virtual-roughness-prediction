from sklearn import linear_model
from sklearn.svm import SVR

from src.model.prediction.compared.cnn_1d.wassem import CNN1D4HA
from src.model.prediction.compared.cnn_1d.scirep import CNN1DScirep
from src.model.prediction.compared.ann import ANN

# from src.model.prediction.proposed.transformer import TransformerRegressor
from src.model.prediction.compared.cnn_1d.generic import CNN1DGeneric
from src.model.prediction.proposed.gated_mlp import GatedFusionRegressor
from src.model.prediction.proposed.gated_mlp_v2 import GatedFusionRegressorV2

MODEL_REGISTRY = {
    "lr": lambda conf, input_dim, device: linear_model.LinearRegression(),
    "svr": lambda conf, input_dim, device: SVR(),
    "ann": lambda conf, input_dim, device: ANN(conf, input_dim=input_dim).to(device),
    "cnn_1d_scirep": lambda conf, input_dim, device: CNN1DScirep(conf, input_dim=input_dim).to(device),
    "cnn_1d_4ha": lambda conf, input_dim, device: CNN1D4HA(conf, input_dim=input_dim).to(device),
    # "transformer": lambda conf, input_dim, device: TransformerRegressor().to(device),
    "cnn_1d_generic": lambda conf, input_dim, device: CNN1DGeneric(input_dim=input_dim).to(device),
    "gated_mlp": lambda conf, input_dim, device: GatedFusionRegressor(input_dim=input_dim).to(device),
    "gated_mlp_v2": lambda conf, input_dim, device: GatedFusionRegressorV2(input_dim=input_dim).to(device),
}

def create_model(conf, input_dim, device=None):
    model_name = conf["model"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](conf, input_dim, device)