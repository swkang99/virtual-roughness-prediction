Project: virtual-tactile-property

Input : Texture PBR maps -> Output : Roughness

Directory
```
virtual-tactile-property/
├── config.yaml
├── README.md
├── requirements.txt
├── train.py
├── val.py
├── data/    # currently ignored. Please get from your PC
│   ├── original/
│   │   ├── adjective_rating_shuffled.csv
│   │   ├── height_map/
│   │   ├── normal_map/
│   │   └── texture_image/
│   └── split/
│       ├── train_ids.csv
│       ├── valid_ids.csv
│       ├── train/
│       │   ├── height_map/
│       │   ├── normal_map/
│       │   └── texture_image/
│       └── valid/
│           ├── height_map/
│           ├── normal_map/
│           └── texture_image/
├── experiments/
│   ├── checkpoints/
│   │   ├── {feature_extractor}/
│   │   │   └── best_model.pth
│   └── runs/
│       ├── {feature_extractor}/
│       │   ├── training_log.csv
│       │   ├── val_metrics.csv
│       │   └── val_summary.txt
└── src/
    ├── __init__.py
    ├── engine/   # about feature extracting, training loop
    │   ├── check_feature_cache.py
    │   ├── engine.py
    │   └── extract_feature.py
    ├── model/    # model definition (class)
    │   └── model.py
    └── utils/
        ├── data.py
        ├── plot.py
        ├── running_time_test.py
        └── split_dataset.py
```