#!/bin/bash
CKPT=llava-uhd-144-13b-loc-unpad-fullft

CUDA_VISIBLE_DEVICES=0 python gradio_demo.py \
    --model-path ./checkpoints_new/$CKPT \
    --fted_encoder True