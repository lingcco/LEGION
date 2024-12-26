#!/bin/bash

export MASTER_PORT=12345


CE_WEIGHTS=(1.0)
DICE_WEIGHTS=(1.0 2.0 0.5)
BCE_WEIGHTS=(1.0 0.5 2.0)


for ce in "${CE_WEIGHTS[@]}"; do
  for dice in "${DICE_WEIGHTS[@]}"; do
    for bce in "${BCE_WEIGHTS[@]}"; do     
      deepspeed --master_port $MASTER_PORT train.py \
        --version '/mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained' \
        --dataset_dir ./data/ \
        --vision_pretrained /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth \
        --exp_name 'Legion' \
        --lora_r 8 \
        --lr 3e-4 \
        --ce_loss_weight $ce \
        --dice_loss_weight $dice \
        --bce_loss_weight $bce \
        --pretrained \
        --use_segm_data \
        --seg_dataset "Legion" \
        --segm_sample_rates "1,3,3,3,1" \
        --val_dataset "Legion" \
        --epochs 5 \
        --batch_size 16 \
        --steps_per_epoch 144 \
        --wandb_log True
        # 如果需要恢复训练，可以取消下面两行的注释并设置相应路径
        # --mask_validation \
        # --resume /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_last_epoch \
        # --start_epoch 5 

      echo "训练完成"
      echo "---------------------------------------------"
    done
  done
done
