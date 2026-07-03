import copy
import yaml

from src.trainer import Trainer
from src.model.factory import create_model


def build_conf(base_conf, model_name):
    conf = copy.deepcopy(base_conf)
    conf["model"] = model_name
    conf["train_tag"] = f"patch_roughness_{model_name}_300epoch"
    return conf


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        base_conf = yaml.safe_load(f)

    model_list = [
        # "lr",
        # "svr",
        # "ann",
        # "cnn_1d_scirep",
        # "cnn_1d_4ha",
        # "transformer",
        "cnn_1d_generic",
        # "gated_mlp",
        # "gated_mlp_v2",
    ]

    trainer = Trainer(conf=copy.deepcopy(base_conf), model_builder=create_model)

    for model_name in model_list:
        conf = build_conf(base_conf, model_name)

        trainer.conf = conf
        trainer.epochs = int(conf.get("epochs", trainer.epochs))
        trainer.batch_size = int(conf.get("batch_size", trainer.batch_size))
        trainer.lr = float(conf.get("learning_rate", trainer.lr))
        trainer.weight_decay = float(conf.get("weight_decay", trainer.weight_decay))
        trainer.seed = conf.get("seed", trainer.seed)
        trainer.verbose = conf.get("verbose", trainer.verbose)
        trainer.set_seed(trainer.seed)

        result = trainer.fit_splits()
        print(f"[{model_name}] metrics: {result['metrics']}")


if __name__ == "__main__":
    main()