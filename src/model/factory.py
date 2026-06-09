from src.model.prediction.compared.cnn_1d_4ha import CNN1D4HA
# from src.model.prediction.compared.cnn_1d_scirep import CNN1DScirep
# from src.model.prediction.compared.ann import ANN
# from src.model.prediction.compared.lr import LinearRegressor
# from src.model.prediction.compared.svr import SVRModel
from src.model.prediction.proposed.transformer import TransformerRegressor

MODEL_REGISTRY = {
    "lr": lambda conf, input_dim, device: LinearRegressor(conf),
    "svr": lambda conf, input_dim, device: SVRModel(conf),
    "ann": lambda conf, input_dim, device: ANN(conf, input_dim=input_dim).to(device),
    "cnn_1d_scirep": lambda conf, input_dim, device: CNN1DScirep(conf, feature_dim=input_dim).to(device),
    "cnn_1d_4ha": lambda conf, input_dim, device: CNN1D4HA(conf).to(device),
    "transformer": lambda conf, input_dim, device: TransformerRegressor().to(device),
}

def create_model(conf, input_dim=None, device=None):
    model_name = conf["model"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](conf, input_dim, device)