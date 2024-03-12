export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export TRAIN_DIR="datasets/ln_sdxl_1k"


accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --enable_xformers_memory_efficient_attention \
  --train_data_dir=$TRAIN_DIR \
  --resolution=1024 --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=10000 \
  --checkpointing_steps 2000 --checkpoints_total_limit 20 \
  --learning_rate=3e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --rank 128 \
  --output_dir="test"
