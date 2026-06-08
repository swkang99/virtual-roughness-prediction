# Implementation of proposed model type B
# Feature Projection : MLP after feature extraction
# Concat
# MLP Fusion (simple than type A) : 
#     1. Linear(d1) + BatchNorm + ReLU
#     2. Linear(1) -> Scalar Output
# MSE Loss
# Adam optimizer
# Weight decay, L2 Regularization, Dropout, ...