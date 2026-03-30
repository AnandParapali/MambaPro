#!/bin/bash

#downloading micromamba for env creation
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

#make mamba directory
mkdir -p /kaggle/working/mamba

#create enviornment
/kaggle/working/bin/micromamba create -y -p /kaggle/working/mamba/envs/MambaPro python=3.10.13

# install PyTorch 2.1.1 with CUDA 11.8
/kaggle/working/bin/micromamba run -p /kaggle/working/mamba/envs/MambaPro \
python -m pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
--index-url https://download.pytorch.org/whl/cu118

#make sure numpy version is less than 
/kaggle/working/bin/micromamba run -p /kaggle/working/mamba/envs/MambaPro \
python -m pip install "numpy<2"

/kaggle/working/bin/micromamba run -p /kaggle/working/mamba/envs/MambaPro \
python -m pip install transformers==4.38.2 timm==0.9.12 einops yacs tqdm h5py scipy==1.12.0 scikit-learn==1.3.2 --no-deps

/kaggle/working/bin/micromamba run -p /kaggle/working/mamba/envs/MambaPro \
python -m pip install opencv-python==4.9.0.80

/kaggle/working/mamba/envs/MambaPro/bin/python -m pip install joblib threadpoolctl

/kaggle/working/bin/micromamba install -p /kaggle/working/mamba/envs/MambaPro \
    -c nvidia/label/cuda-11.8.0 \
    cuda-toolkit cuda-nvcc -y

# Define the python path
env_python="/kaggle/working/mamba/envs/MambaPro/bin/python"

$env_python -m pip install "setuptools<70.0.0"

$env_python -m pip install causal-conv1d==1.1.1 --no-build-isolation

# Install requirements WITHOUT constraints
# This will allow pip to update versions if needed, but will build using 11.8
$env_python -m pip install -r requirements.txt --no-build-isolation --upgrade

# We use the full path to your environment's pip to ensure it goes to the right place
/kaggle/working/mamba/envs/MambaPro/bin/pip install git+https://github.com/openai/CLIP.git

echo "Setup Complete! Environment is ready at /kaggle/working/mamba/envs/MambaPro"  
