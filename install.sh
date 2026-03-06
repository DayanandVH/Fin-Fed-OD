#!/bin/bash

apt update ; apt install pip; apt clean

# apt install python3.7

# pip install flwr
# pip install flwr[simulation]

# pip install flwr==1.3.0
pip install flwr[simulation]==1.3.0

# pip install flwr==1.0.0
# # pip install torch==1.12.0
# pip install torchvision==0.13.0

# !pip install -q flwr[simulation]
# pip install -U flwr["simulation"]

# pip install numpy==1.15.0
# pip install numpy==1.8

# pip install scipy==1.7.0
# pip install scipy==1.5.1

# apt install python3.7.6
pip install pyod
# pip install keras==2.3.1
pip install seaborn
pip install -U ray==2.2.0
pip install grpcio==1.41.0
pip install gpustat==1.0

# pip uninstall grpcio
# conda install grpcio

# python3 -u fl_clients_latent_parameter_sharing_all_type_outlier_prediction_bce_and_mse_loss.py 5
