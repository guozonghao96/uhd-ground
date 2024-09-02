#!/bin/bash
export CUDA_HOME=/home/test/test08/gzh/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


CKPT=llava-uhd-144-13b-loc-unpad
OUTPUT_DIR=./checkpoints_new/$CKPT
LLM_CKPT_DIR=./pretrained_models/vicuna-13b-v1.5
CLIP_CKPT_DIR=./pretrained_models/clip-vit-large-patch14-336
echo $OUTPUT_DIR


# pretraining script
# PRBATCH=32
# ACCU_STEPS=1
# deepspeed \
#     --master_port=12322 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path $LLM_CKPT_DIR \
#     --version plain \
#     --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder ./playground/data/LLaVA-Pretrain/images \
#     --vision_tower $CLIP_CKPT_DIR \
#     --mm_projector_type adapt_spatial_resampler \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $PRBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $ACCU_STEPS \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --single True


    # --data_path ./playground/data/llava_v1_5_mix665k.json \

# full ft script
# FTRBATCH=4
# ACCU_STEPS=4
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path $LLM_CKPT_DIR \
#     --version v1 \
#     --data_path ./playground/data/updated_llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower $CLIP_CKPT_DIR \
#     --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
#     --mm_projector_type adapt_spatial_resampler \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $FTRBATCH \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $ACCU_STEPS \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

    #  \
    # --tune_mm_mlp_adapter True \
    # --mm_use_loc_token True \
    # --pretrain_word_embedding False \
    # --tune_word_embedding True

    # stage 1: tune_mm_mlp_adapter=True, mm_use_loc_token=False 
    # stage 2: tune_mm_mlp_adapter=True, mm_use_loc_token=True（需要增加loc token的训练），pretrain_word_embedding=False（还没pretrain这些embedding），tune_word_embedding=True（只Tune这些embedding）
    # stage 3: tune_mm_mlp_adapter=False(打开LLM训练), mm_use_loc_token=True（需要增加loc token的训练），pretrain_word_embedding=True（用的是stage2的mm_projector.bin中存的参数），tune_word_embedding=False/True（因为LLM都tune了，不用专门调True/False）
        # stage 2 之后vocab_size从32000到了32004，存在在checkpoint_new中的config.json，但是stage 3还是从vicuna那边去load config.json，所以还是32000，因此还需要再过一边初始化。这里面load的操作有两种。
            # 第一种：只load新增的token的word embedding和lm_head的（目前这套代码）
            # 第二种：全load这些embedding
    # inference： 这些参数目前都没有写到config里面，但是vocab_size增加好，然后保存的参数也都是训练好的，因此直接inference

# evaluation

    # --data_path ./playground/data/llava_UHD_sft_mix932k.json \
    # --data_path ./playground/data/updated_llava_v1_5_mix665k.json \
    # --include localhost:0 \
FTRBATCH=8
ACCU_STEPS=2

deepspeed \
    --master_port=12302 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version v1 \
    --data_path ./playground/data/llava_UHD_sft_mix932k.json \
    --image_folder ./playground/data \
    --vision_tower $CLIP_CKPT_DIR \
    --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
    --mm_projector_type adapt_spatial_resampler \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $FTRBATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --tune_mm_mlp_adapter False \
    --mm_use_loc_token True \
    --pretrain_word_embedding True \
    --tune_word_embedding True


# sh eval.sh $CKPT
