## ChatCAD_R

This code is created and modified based on [ChatCAD](https://github.com/zhaozh10/ChatCAD) by Wang et al.

We edited certain prompts and pipelines to fit our tasks, and created util functions for more convenient function calls. We also removed revChatGPT and replaced it by openai v1. 

The code implements retrieval augmented generation (RAG) functionality on three main tasks:

1. General medical advice generation
2. Chest X-ray image report analysis
3. General medical report analysis

Example use of the three functionalities are implemented in [test.ipynb](test.ipynb).

The scripts to generate thinking chain conversation JSON files of the three functionalities for LLM training are implemented in

1. [RAG_gen_casual.py](RAG_gen_casual.py)
2. [RAG_gen_imgreport_1.py](RAG_gen_imgreport_1.py) and [RAG_gen_imgreport_2.py](RAG_gen_imgreport_2.py)
3. [RAG_gen_textreport.py](RAG_gen_textreport.py)

*If proxy is needed to access GPT from certain regions, refer to comments and edit correspondingly in [chat_bot_RAG.py](chat_bot_RAG.py) and [engine_LLM\api.py](engine_LLM/api.py).*
