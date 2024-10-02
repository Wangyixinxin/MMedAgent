CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
    --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/llava_ft_vqatool \
    --question-file \
    ./rad_vqa_test.jsonl \
    --image-folder \
    ./yixin-llm/vqa_rad_test \
    --answers-file \
   ./Rebuttal_Eval/llava_vqa_tool_ans.jsonl\
    --temperature 0.2
