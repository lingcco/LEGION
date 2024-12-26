#!/bin/bash

tag='global_step686'
python /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_best/zero_to_fp32.py \
    /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_best \
    /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_best/$tag/pytorch_model.bin \
    --tag "$tag"
 