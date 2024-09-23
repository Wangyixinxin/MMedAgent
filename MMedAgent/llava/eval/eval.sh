CUDA_VISIBLE_DEVICE=0 python model_vqa.py \
    --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/llava_plus_v0_7b\
    --question-file /home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/llava/eval/eval.jsonl \
    --image-folder /home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/llava/eval/img\
    --answers-file ./eval_ans.jsonl