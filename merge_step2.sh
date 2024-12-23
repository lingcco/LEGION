export PYTHONPATH="./:$PYTHONPATH"
tag='global_step686'

python scripts/merge_lora_weights.py \
    --version /mnt/hwfile/opendatalab/wensiwei/checkpoint/GLaMM-GranD-Pretrained \
    --weight /mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/ckpt_model_best/$tag/pytorch_model.bin \
    --save_path /mnt/petrelfs/wensiwei/LEGION/groundingLMM/checkpoint/$tag \
    --vision_pretrained  /mnt/hwfile/opendatalab/wensiwei/checkpoint/SAM/sam_vit_h_4b8939.pth 
