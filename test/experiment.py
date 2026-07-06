import copy
import yaml
import argparse

from src.trainer import Trainer
from src.model.factory import create_model


def build_conf(base_conf, model_name):
    conf = copy.deepcopy(base_conf)
    conf["model"] = model_name
    conf["train_tag"] = f"patch_roughness_{model_name}_300epoch"
    return conf


def run_models(trainer, base_conf, model_list, args=None):
    """Run multi-model training with optional CLI argument overrides."""
    results = {}
    for model_name in model_list:
        conf = build_conf(base_conf, model_name)
        trainer.conf = conf

        # Use CLI args if provided, else fall back to config values
        if args:
            trainer.epochs = args.epochs if args.epochs is not None else int(conf.get("epochs", trainer.epochs))
            trainer.batch_size = args.batch_size if args.batch_size is not None else int(conf.get("batch_size", trainer.batch_size))
            trainer.lr = args.learning_rate if args.learning_rate is not None else float(conf.get("learning_rate", trainer.lr))
            trainer.weight_decay = args.weight_decay if args.weight_decay is not None else float(conf.get("weight_decay", trainer.weight_decay))
            trainer.seed = args.seed if args.seed is not None else conf.get("seed", trainer.seed)
            trainer.verbose = args.verbose if args.verbose is not None else conf.get("verbose", trainer.verbose)
        else:
            trainer.epochs = int(conf.get("epochs", trainer.epochs))
            trainer.batch_size = int(conf.get("batch_size", trainer.batch_size))
            trainer.lr = float(conf.get("learning_rate", trainer.lr))
            trainer.weight_decay = float(conf.get("weight_decay", trainer.weight_decay))
            trainer.seed = conf.get("seed", trainer.seed)
            trainer.verbose = conf.get("verbose", trainer.verbose)

        trainer.set_seed(trainer.seed)
        result = trainer.fit_splits()
        results[model_name] = result
        print(f"[{model_name}] metrics: {result['metrics']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train multiple models with optional config overrides.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config.yaml)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config.yaml)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config.yaml)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config.yaml)")
    parser.add_argument("--model", type=str, default=None, help="Train specific model only (e.g., transformer)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        base_conf = yaml.safe_load(f)

    if args.model:
        model_list = [args.model]
    else:
        model_list = ["lr", "svr", "ann", "cnn_1d_scirep", "cnn_1d_4ha", "transformer", "cnn_1d_generic", "gated_mlp", "gated_mlp_v2"]

    trainer = Trainer(conf=copy.deepcopy(base_conf), model_builder=create_model)
    run_models(trainer, base_conf, model_list, args)


if __name__ == "__main__":
    main()