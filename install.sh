#!/bin/bash
export CUDA_HOME=/home/test/test08/gzh/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# pip install imgaug
# pip install openpyxl
# pip install gpustat 

# pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation


# wandb offline
