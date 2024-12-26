export MASTER_PORT=12345

deepspeed --master_port $MASTER_PORT train.py \
  --version '/mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained' \
  --dataset_dir ./data/ \
  --vision_pretrained /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth \
  --exp_name 'Legion' \
  --lora_r 8 \
  --lr 3e-4 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 1.0 \
  --bce_loss_weight 1.0 \
  --pretrained \
  --use_segm_data \
  --seg_dataset "Legion" \
  --segm_sample_rates "1,3,3,3,1" \
  --val_dataset "Legion" \
  --epochs 5 \
  --batch_size 16 \
  --steps_per_epoch 144 \
  --wandb_log True
  # --mask_validation \
  # --resume /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_last_epoch \
  # --start_epoch 5 
  #后两行是resume才有