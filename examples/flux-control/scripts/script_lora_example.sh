#!/bin/bash
# export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
# export INSTANCE_DIR="dog"
# export MASK_DIR="dog_masks"
# export OUTPUT_DIR="flux-fill-dog-lora"

accelerate launch train_control_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control-lora" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_image="openpose.png" \
  --validation_prompt="A couple, 4k photo, highly detailed" \
  --offload \
  --seed="0"