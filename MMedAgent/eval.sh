# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
#     --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/ft_500_llava_2e5 \
#     --question-file \
#     /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_agent_plus_toy.jsonl \
#     --image-folder \
#     /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file \
#    /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_llavamed_plus_ans_toy_new_2e5_500.jsonl\
#     --temperature 0.2

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa_34b.py \
#     --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/llava-v1.6-34b \
#     --question-file \
#     /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_agent.jsonl\
#     --image-folder \
#     /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file \
#    /home/jack/Projects/yixin-llm/Rebuttal_Eval/llava34b.jsonl\
#     --temperature 0.2

CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
    --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/llava_ft_vqatool \
    --question-file \
    /home/jack/Projects/yixin-llm/rad_vqa_test.jsonl \
    --image-folder \
    /home/jack/Projects/yixin-llm/vqa_rad_test \
    --answers-file \
   /home/jack/Projects/yixin-llm/Rebuttal_Eval/llava_vqa_tool_ans.jsonl\
    --temperature 0.2

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
#     --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/ft_500_llava \
#     --question-file \
#     /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_agent_plus_toy.jsonl \
#     --image-folder \
#     /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file \
#    /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_llavamed_plus_ans_toy_new_500.jsonl\
#     --temperature 0.2

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
#     --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/ft_500_llava \
#     --question-file \
#     /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_agent_plus_toy.jsonl \
#     --image-folder \
#     /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file \
#    /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_llavamed_plus_ans_toy_new_500.jsonl\
#     --temperature 0.2



# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa_1stage.py \
#     --model-path /home/jack/Projects/yixin-llm/yixin-llm-data/ft_1500_llava \
#     --question-file \
#     /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_agent_plus_toy.jsonl \
#     --image-folder \
#     /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file \
#    /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_llavamed_plus_ans_toy_new_1500.jsonl\
#     --temperature 0.2


# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
#     --model-path /home/jack/Projects/yixin-llm/merge_med_llava_3 \
#     --question-file /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_193_plus.jsonl \
#     --image-folder /home/jack/Projects/yixin-llm/eval_images \
#     --answers-file /home/jack/Projects/yixin-llm/LLaVA-EVAL/qa_base_193_mmedagent.jsonl \
#     --temperature 0.2