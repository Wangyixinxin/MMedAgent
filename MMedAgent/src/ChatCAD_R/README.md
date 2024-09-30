## ChatCAD_R

This code is based on [ChatCAD](https://github.com/zhaozh10/ChatCAD) by Wang et al.

The code implements retrieval augmented generation (RAG) functionality on three main tasks:

1. General medical advice generation
2. Chest X-ray image report analysis
3. General medical report analysis

Example use of the three functionalities are implemented in [test.ipynb](https://github.com/Wangyixinxin/MMedAgent/blob/main/ChatCAD_R/test.ipynb).

The scripts to generate thinking chain conversation JSON files of the three functionalities for LLM training are implemented in

1. [RAG_gen_casual.py](https://github.com/Wangyixinxin/MMedAgent/blob/main/ChatCAD_R/RAG_gen_casual.py)
2. [RAG_gen_imgreport_1.py](https://github.com/Wangyixinxin/MMedAgent/blob/main/ChatCAD_R/RAG_gen_imgreport_1.py) and [RAG_gen_imgreport_2.py](https://github.com/Wangyixinxin/MMedAgent/blob/main/ChatCAD_R/RAG_gen_imgreport_2.py)
3. [RAG_gen_textreport.py](https://github.com/Wangyixinxin/MMedAgent/blob/main/ChatCAD_R/RAG_gen_textreport.py)

   