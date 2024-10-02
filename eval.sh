CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa.py \
    --model-path ./llava_med_agent \
    --question-file ./eval_data_json/eval_example.jsonl \
    --image-folder ./eval_images \
    --answers-file ./eval_data_json/output_agent_eval_example.jsonl \
    --temperature 0.2