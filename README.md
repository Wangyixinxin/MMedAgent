# MMedAgent: Learning to Use Medical Tools with Multi-modal Agent

The first multimodal medical AI Agent incorporating a wide spectrum of tools to handle various medical
tasks across different modalities seamlessly.

[[Paper, EMNLP 2024 (Findings)](https://arxiv.org/abs/2407.02483)] [Demo]

Binxu Li, Tiankai Yan, Yuanting Pan, Zhe Xu, Jie Luo, Ruiyang Ji, Shilong Liu, Haoyu Dong*, Zihao Lin*, Yixin Wang* 

<div style="text-align: center;">
    <img src="imgs/mmedagent.jpg" alt="MMedAgent" style="width: 50%;"/>
    <img src="imgs/instruction-tuning-data.jpg" alt="Instruction Tuning Data" style="width: 50%;"/>
</div>

## Current Tool lists
(1) grounding, (2) segmentation with bounding-box prompts (Segmentation),
(3) segmentation with text prompts (G-Seg), (4)
medical imaging classification, (5) Medical Report
Generation (MRG), (6) retrieval augmented generation (RAG), and (7) VQA.
| task | tool name | link |
| --- | --- | --- |
| Detection | grounding dino (medical version) | <https://github.com/IDEA-Research/GroundingDINO> |
| Segmentation with bounding-box prompts (Segmentation) | MedSAM | <https://github.com/bowang-lab/MedSAM> |
| segmentation with text prompts (G-Seg) | grounding dino + MedSAM | <https://github.com/bowang-lab/MedSAM> |
| Medical imaging classification | Biomedclip | <https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224> |
| Medical report generation (MRG) | ChatCAD-G | <https://github.com/zhaozh10/ChatCAD> |
| Retrieval augmented generation (RAG) | ChatCAD-R | <https://github.com/zhaozh10/ChatCAD> |

| Task           | Tool                                     | Data Source                                                                                                                       | Imaging Modality                             |
|----------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| VQA            | LLaVA-Med                    | PMC article<br>*60K-IM*                                                                                                | MRI, CT, X-ray, Histology, Gross            |
| Classification | BiomedCLIP                       | PMC article<br>*60K-IM*                                                                                                         | MRI, CT, X-ray, Histology, Gross            |
| Grounding      | Grounding DINO                    | WORD, etc.*<br>                                                                                                                 | MRI, CT, X-ray, Histology                   |
| Segmentation with bounding-box prompts (Segmentation)    | MedSAM                            | WORD, etc.*                                                                                                                      | MRI, CT, X-ray, Histology, Gross            |
| Segmentation with text prompts (G-Seg)        | Grounding DINO + MedSAM                  | WORD, etc.*                                                                                                                      | MRI, CT, X-ray, Histology                   |
| Medical report generation (MRG)            | ChatCAD                           | MIMIC-CXR                                                                                                               | X-ray                                        |
| Retrieval augmented generation (RAG)            | ChatCAD+                         | Merck Manual                                                                                                            | --                                           |

---

**Note**: ``--`` means that the RAG task only focuses on natural language without handling images. ``WORD, etc.*`` indicates various data sources including WORD, FLARE2021, BRATS, Montgomery County X-ray Set (MC), VinDr-CXR, and Cellseg.  


## Usage
1. Clone this repo and navigate to xxx folder
```
xx
```
2. Run..

## Evaluation

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
