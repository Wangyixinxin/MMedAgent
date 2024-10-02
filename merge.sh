CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/output_lora_weights \
    --model-base ./base_model \
    --save-model-path ./llava_med_agent