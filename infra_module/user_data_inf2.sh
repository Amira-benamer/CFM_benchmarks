#!/bin/bash

set -e  # Exit on error

# Update and install required packages
sudo apt-get update -y
sudo apt-get install -y python3-venv linux-headers-$(uname -r) curl

# Add Neuron APT repository (Ubuntu 20.04 "focal")
echo "deb https://apt.repos.neuron.amazonaws.com focal main" | sudo tee /etc/apt/sources.list.d/neuron.list
curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update package list
sudo apt-get update -y

# Install Neuron components
sudo apt-get install -y --allow-change-held-packages \
    aws-neuronx-dkms=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-collectives=2.* \
    aws-neuronx-tools=2.*

# Create Python virtual environment
python3 -m venv /opt/neuron-env

# Activate the environment
source /opt/neuron-env/bin/activate

# Upgrade pip and install required Python packages
pip install --upgrade pip
pip install 'torch>=1.8' torch-neuron torchvision neuron-cc[tensorflow]

# Add Neuron tools to PATH for all future sessions
echo 'export PATH=/opt/aws/neuron/bin:$PATH' >> /etc/profile
echo 'source /opt/neuron-env/bin/activate' >> /etc/profile
export PATH=/opt/aws/neuron/bin:$PATH
