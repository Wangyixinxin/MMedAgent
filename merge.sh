CUDA_VISIBLE_DEVICES=1 python scripts/merge_lora_weights.py \
    --model-path /home/jack/Projects/yixin-llm/LLaVA-Plus/checkpoints/medagent_vqa_finetune\
    --model-base /home/jack/Projects/yixin-llm/merge_med_llava_2 \
    --save-model-path /home/jack/Projects/yixin-llm/yixin-llm-data/llava_ft_vqatool