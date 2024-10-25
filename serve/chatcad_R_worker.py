"""
A model worker executes the model.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "ChatCAD_R"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util import *

import argparse
import asyncio
import os
import sys
import time
from typing import List, Tuple, Union
import threading
import uuid
import torchvision

from io import BytesIO
import base64

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import requests
from PIL import Image
import uvicorn
import openai

from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from serve.utils import build_logger, pretty_print_semaphore

GB = 1 << 30


now_file_name = os.__file__
logdir = "logs/workers/"
os.makedirs(logdir, exist_ok=True)
logfile = os.path.join(logdir, f"{now_file_name}.log")

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(now_file_name, logfile)
global_counter = 0

model_semaphore = None

# If building your own server, you can initialize chatbot in advance to acclerate inference speed on server
# chatbot = initialize_chatbot("")  # put your api key
chatbot = None

def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_names
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()



    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }


    def generate_stream_func(self, params):
        report = RAG(params["prompt"], api_key=params["openai_key"],chatbot=chatbot)
        return report


    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(
                params
            )
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    openai.api_key = params["openai_key"]
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/model_details")
async def model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21021)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21021")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument(
        "--model-names",
        default="ChatCAD-R",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_names,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
