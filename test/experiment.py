import copy
import yaml

from src.loocv import loocv
from src.model.factory import create_model

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        base_conf = yaml.safe_load(f)

    model_list = [
        # "lr",
        # "svr",
        # "ann",
        "cnn_1d_scirep",
        "cnn_1d_4ha",
        "transformer",
    ]

    for model_name in model_list:
        conf = copy.deepcopy(base_conf)
        conf["model"] = model_name

        print(f"\n===== Running LOOCV for {model_name} =====")
        loocv(conf, model_builder=create_model)

if __name__ == "__main__":
    main()