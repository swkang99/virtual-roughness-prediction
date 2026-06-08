# Implementation of proposed model type A
# Concat Features : texture image + height map + normal map
# MLP Fusion : 
#     1. Linear(d1) + BatchNorm + ReLU
#     2. Linear(d2) + BatchNorm + ReLU
#     3. Linear(1) -> Scalar Output
# MSE Loss
# Adam optimizer
# Weight decay, L2 Regularization, Dropout, ...