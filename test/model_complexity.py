import torch
from ptflops import get_model_complexity_info

model = ...  # your nn.Module
model.eval()

with torch.cuda.device(0):
    flops, params = get_model_complexity_info(
        model,
        (3, 224, 224),
        as_strings=False,
        print_per_layer_stat=False
    )

print("FLOPs:", flops)
print("Params:", params)