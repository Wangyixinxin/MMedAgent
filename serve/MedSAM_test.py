import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from MedSAM_utils import *

import argparse
import os
import sys
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import requests
import time


def load_image(image_path: str):
    img = Image.open(image_path).convert('RGB')
    return img

def encode(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def main():
    model_name = args.model_name
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return
    
    headers = {"User-Agent": "FastChat Client"}
    image = load_image(args.image_path)
    image = encode(image)
    datas = {
        "model": model_name,
        "image": image,
    }
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=datas,
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")
    print("ChatCAD-G result:")
    print(response)
    print(response.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='MedSAM')

    # model parameters
    parser.add_argument(
        "--image_path", type=str, default="/home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/src/MedSAM/assets/img_demo.png"
    )
    parser.add_argument(
        "--save_path", type=str, default="/home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/src/MedSAM/assets/"
    )
    parser.add_argument(
        "--bbox", type=str, default="[95, 255, 190, 350]"
    )
    parser.add_argument(
        "--ckpt", type=str, default="/home/jack/Projects/yixin-llm/yixin-llm-data/yptests/LLaVA-Plus/src/MedSAM/medsam_vit_b.pth"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0"
    )
    args = parser.parse_args()
    main()