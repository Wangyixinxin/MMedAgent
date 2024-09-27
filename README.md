# MMedAgent: Learning to Use Medical Tools with Multi-modal Agent

The first multimodal medical AI Agent incorporating a wide spectrum of tools to handle various medical
tasks across different modalities seamlessly.

[[Paper, EMNLP 2024 (Findings)](https://arxiv.org/abs/2407.02483)] [Demo]

Binxu Li, Tiankai Yan, Yuanting Pan, Jie Luo, Ruiyang Ji, Jiayuan Ding, Zhe Xu, Shilong Liu, Haoyu Dong*, Zihao Lin*, Yixin Wang* 

<div style="text-align: center;">
    <img src="imgs/mmedagent.jpg" alt="MMedAgent" style="width: 50%;"/>
    <img src="imgs/instruction-tuning-data.jpg" alt="Instruction Tuning Data" style="width: 50%;"/>
</div>

## Current Tool lists

| Task           | Tool                                     | Data Source                                                                                                                       | Imaging Modality                             |
|----------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| VQA            | [LLaVA-Med](https://github.com/microsoft/LLaVA-Med/tree/main)                    | PMC article<br>*60K-IM*                                                                                                | MRI, CT, X-ray, Histology, Gross            |
| Classification | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)                       | PMC article<br>*60K-IM*                                                                                                         | MRI, CT, X-ray, Histology, Gross            |
| Grounding      | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)                    | WORD, etc.*<br>                                                                                                                 | MRI, CT, X-ray, Histology                   |
| Segmentation with bounding-box prompts (Segmentation)    | [MedSAM](https://github.com/bowang-lab/MedSAM)                            | WORD, etc.*                                                                                                                      | MRI, CT, X-ray, Histology, Gross            |
| Segmentation with text prompts (G-Seg)        | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)  + [MedSAM](https://github.com/bowang-lab/MedSAM)                  | WORD, etc.*                                                                                                                      | MRI, CT, X-ray, Histology                   |
| Medical report generation (MRG)            | [ChatCAD](https://github.com/zhaozh10/ChatCAD)                           | MIMIC-CXR                                                                                                               | X-ray                                        |
| Retrieval augmented generation (RAG)            | [ChatCAD+](https://github.com/zhaozh10/ChatCAD)                         | Merck Manual                                                                                                            | --                                           |

---

**Note**: ``--`` means that the RAG task only focuses on natural language without handling images. ``WORD, etc.*`` indicates various data sources including WORD, FLARE2021, BRATS, Montgomery County X-ray Set (MC), VinDr-CXR, and Cellseg.  


## Usage
1. Clone this repo and navigate to xxx folder
```
git clone https://github.com/Wangyixinxin/MMedAgent.git
```
2. Run..
## Train
```
deepspeed llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /path_to_your_basemodel  \
    --version v1\
    --data_path /path_to_data \
    --image_folder /home/jack/Projects/yixin-llm/images_path/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/medagentv6 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```
## Evaluation
### apply lora (if you enable lora when training)
```
CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path /path_to_lora_weights \
    --model-base /path_to_base_model \
    --save-model-path /path_to_model_checkpoint
```
### Inference
```
CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa.py \
    --model-path /path_to_your_model \
    --question-file /path_to_your_eval_data \
    --image-folder /path_to_your_eval_image \
    --answers-file /path_to_your_output_answer \
    --temperature 0.2
```
## Data Download
### Instruction-tuning Dataset
We build the first open-source instruction tuning dataset for multi-modal medical agents.

| Data | size |
| --- | --- |
| xxx | xx MiB | 

### Tool dataset (Selected)

#### Grounding task dataset
| Data | size |
| --- | --- |
| xxx | xx MiB | 
#### Segmentation task dataset
| Data | size |
| --- | --- |
| xxx | xx MiB | 

## Model Download

## Web UI


## Citation
If you find this paper or code useful for your research, please cite our paper:
```
@article{li2024mmedagent,
  title={MMedAgent: Learning to Use Medical Tools with Multi-modal Agent},
  author={Li, Binxu and Yan, Tiankai and Pan, Yuanting and Xu, Zhe and Luo, Jie and Ji, Ruiyang and Liu, Shilong and Dong, Haoyu and Lin, Zihao and Wang, Yixin},
  journal={arXiv preprint arXiv:2407.02483},
  year={2024}
}
```
## Related Project
MMedAgent was built on [LLaVA-PLUS](https://llava-vl.github.io/llava-plus/) and [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) was chosen as the backbone. 

## Contributing
We are working on extending the current tool lists to handle more medical tasks and modalities. We deeply appreciate any contribution made to improve the our Medical Agent. If you are developing better LLM-tools, feel free to contact us!
