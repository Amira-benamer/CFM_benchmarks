#!/bin/bash
exec > /home/ubuntu/user_data.log 2>&1

# Update and install system packages
sudo apt-get update
sudo apt-get install -y python3-pip git

# Activate PyTorch environment
source /opt/pytorch/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision transformers psutil numpy pandas boto3 requests


