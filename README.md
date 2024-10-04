# MMedAgent: Learning to Use Medical Tools with Multi-modal Agent

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Data License](https://img.shields.io/badge/Data%20and%20Weights%20License-CC%20By%20NC%204.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The first multimodal medical AI Agent incorporating a wide spectrum of tools to handle various medical
tasks across different modalities seamlessly.

[[Paper, EMNLP 2024 (Findings)](https://arxiv.org/abs/2407.02483)] [[Demo](https://1cc0bf26516bc745fd.gradio.live/)  (*NOTE: This is a temporary link. Please follow [Web UI Inference](https://github.com/Wangyixinxin/MMedAgent?tab=readme-ov-file#web-ui-and-server) to build the server. *)]

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
1. Clone this repo
```
git clone https://github.com/Wangyixinxin/MMedAgent.git
```
2.  Create environment
```
cd MMedAgent
conda create -n mmedagent python=3.10 -y
conda activate mmedagent
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Additional package for training
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Model Download

### MMedAgent Checkpoint
Model checkpoints (lora) and instruction are uploaded to huggingface: [https://huggingface.co/andy0207/mmedagent]

Download the model and data by following:
```
git lfs install
git clone https://huggingface.co/andy0207/mmedagent
```

### Base Model Download

The model weights below are *delta* weights. The usage of LLaVA-Med checkpoints should comply with the base LLM's model license: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

The delta weights for LLaVA-Med are provided. Download by the following instruction or see details via (LLaVA-Med)[https://github.com/microsoft/LLaVA-Med/tree/v1.0.0]

 Model Descriptions | Model Delta Weights | Size |
| --- | --- | ---: |
| LLaVA-Med | [llava_med_in_text_60k_ckpt2_delta.zip](https://hanoverprod.z21.web.core.windows.net/med_llava/models/llava_med_in_text_60k_ckpt2_delta.zip) | 11.06 GB |

Instructions:

1. Download the delta weights above and unzip.
1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
1. Use the following scripts to get original LLaVA-Med weights by applying our delta. In the script below, set the --delta argument to the path of the unzipped delta weights directory from step 1.

```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target ./base_model \
    --delta /path/to/llava_med_delta_weights
```
## Train
train with lora:
```
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./base_model  \
    --version v1\
    --data_path ./train_data_json/example.jsonl \
    --image_folder ./train_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/output_lora_weights \
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
or use [`tuning.sh`](https://github.com/Wangyixinxin/MMedAgent/blob/main/tuning.sh)
## Evaluation
### Apply lora (if you enable lora during training)
```
CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/output_lora_weights \
    --model-base ./base_model \
    --save-model-path ./llava_med_agent
```
or use [`merge.sh`](https://github.com/Wangyixinxin/MMedAgent/blob/main/merge.sh)
### Inference
```
CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa.py \
    --model-path ./llava_med_agent \
    --question-file ./eval_data_json/eval_example.jsonl \
    --image-folder ./eval_images \
    --answers-file ./eval_data_json/output_agent_eval_example.jsonl \
    --temperature 0.2
```
or use [`eval.sh`](https://github.com/Wangyixinxin/MMedAgent/blob/main/eval.sh)
### GPT-4o inference
```
python llava/eval/eval_gpt4o.py \
    --api-key "your-api-key" \
    --question ./eval_data_json/eval_example.jsonl \
    --output ./eval_data_json/output_gpt4o_eval_example.jsonl \
    --max-tokens 1024
```
or use [`eval_gpt4o.sh`](https://github.com/Wangyixinxin/MMedAgent/blob/main/eval_gpt4o.sh)
### GPT-4 evaluation
All the outputs will be assessed by GPT-4 and rated on a scale from 1 to 10 based on their helpfulness, relevance, accuracy, and level of details. Check our paper for detailed evaluation.
```
python ./llava/eval/eval_multimodal_chat_gpt_score.py \
    --question_input_path ./eval_data_json/eval_example.jsonl \
    --input_path ./eval_data_json/output_gpt4o_eval_example.jsonl
    --output_path ./eval_data_json/compare_gpt4o_medagent_reivew.jsonl
```
or use [`eval_gpt4.sh`](https://github.com/Wangyixinxin/MMedAgent/blob/main/eval_gpt4.sh)
## Data Download
### Instruction-tuning Dataset
We build the first open-source instruction tuning dataset for multi-modal medical agents.

| Data | size |
| --- | --- |
| [instruction_all.json](https://huggingface.co/andy0207/mmedagent/tree/main/instruction_data) | 97.03 MiB | 

Download the data by following:
```
git lfs install
git clone https://huggingface.co/andy0207/mmedagent
```
### Tool dataset (Selected)

#### Grounding task dataset
Please download the following segmentation dataset run the following command to process all data into required data format.
path_writing.py is to write paths for nii file of images and labels into image.txt and label.txt
dataset_loading.py, with stores image.txt and label.txt, is to store jpg format images into a folder, with instances.json file storing the information of images and coodinates of grounding boxes
data_process_func is the helper function from https://github.com/openmedlab/MedLSAM
When use it, please modify the path in the files.

#### Segmentation task dataset
[WORD](https://arxiv.org/pdf/2111.02403
), [FLARE2021](https://flare22.grand-challenge.org/
), [BRATS](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data), [Montgomery County X-ray Set (MC)](), [VinDr-CXR](https://vindr.ai/datasets/cxr
), and [Cellseg](https://zenodo.org/records/10719375)

#### Classification, VQA task dataset
[PMC article 60K-IM](https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json)

## Web UI and serve
1. Download ChatCAD Dependencies

- Please download the dependent checkpoints and JSON files for [src/ChatCAD_R](src/ChatCAD_R).

- You can download from either the original ChatCAD [repo](https://github.com/zhaozh10/ChatCAD?tab=readme-ov-file) or from [Google Drive](https://drive.google.com/drive/folders/14OWwsFjphsjqT-nH9GHgf5Sy7f1aL9Lz?usp=sharing).
  
- Please save r2gcmn_mimic-cxr.pth and JFchexpert.pth in ChatCAD_R/weights/ and save annotation.json in ChatCAD_R/r2g/.

2. Download Grounding Dino Med Checkpoint
- Please download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/1eK27gz0tkbcp-9hx2fI9J_Wj2zHv14pL?usp=sharing).
- Save groundingdinomed-checkpoint0005_slim.pth in [src/](src/)
3. Download Tools
   
   - GroundingDINO
    ```
    cd src
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    ```
   - MedSAM
    ```
    cd src
    git clone https://github.com/bowang-lab/MedSAM.git
    ```

4. Run the following commands in separate terminals:

  - Launch controller
    ```
    python -m llava.serve.controller --host 0.0.0.0 --port 20001
    ```
  - Launch model worker
    ```
    python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:20001 --port 40000 --worker http://localhost:40000 --model-path <Your Model Path>
    ```
  - Launch tool workers
    ```
    python serve/grounding_dino_worker.py
    python serve/MedSAM_worker.py
    python serve/grounded_medsam_worker.py
    python serve/biomedclip_worker.py
    python serve/chatcad_G_worker.py
    python serve/chatcad_R_worker.py    
    ```
  - Launch gradio web server
    ```
    python llava/serve/gradio_web_server_mmedagent.py --controller http://localhost:20001 --model-list-mode reload
    ```
5. You can now access the model in localhost:7860
   
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
