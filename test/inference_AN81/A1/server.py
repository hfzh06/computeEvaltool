# import io
# import asyncio
# import logging
# import time
# import os
# import uvicorn
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# from fastapi import UploadFile, File, Query

# import torch
# from torchvision.models import resnet18, ResNet18_Weights
# from transformers import ViTImageProcessor, ViTForImageClassification
# from ultralytics import YOLO
# import numpy as np

# from PIL import Image

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger('api_server')

# # os.environ['CUDA_VISIBLE_DEVICES']="0"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VIT_HF_ID = "google/vit-large-patch16-224-in21k"
# YOLO_WEIGHTS_ID = "yolov10s.pt"

# YOLO_DEVICE_ARG = DEVICE.index if DEVICE.type == "cuda" else "cpu"
# YOLO_MODEL = YOLO(YOLO_WEIGHTS_ID).to(YOLO_DEVICE_ARG)

# # REPEATS = 32  # number of times to repeat the image in the batch dimension

# MODEL_REGISTRY = {
#     "resnet18": {
#         "kind": "cls_torch",
#         "model": resnet18(weights=ResNet18_Weights.DEFAULT).to(DEVICE).eval(),
#         "preprocess": ResNet18_Weights.DEFAULT.transforms(),
#         "repeat": 256,
#     },
#     "vitlarge": {
#         "kind": "cls_hf",
#         "model": ViTForImageClassification.from_pretrained(VIT_HF_ID, local_files_only=True).to(DEVICE).eval(),
#         "processor": ViTImageProcessor.from_pretrained(VIT_HF_ID, local_files_only=True),
#         "repeat": 8,
#     },
#     "yolov10-s": {
#         "kind": "det",
#         "model": YOLO_MODEL,
#         "repeat": 32,
#         "device_arg": YOLO_DEVICE_ARG,
#     },
# }

# # 281.0835163087285 / 128
# # 351.35016059395025 / 256 
# # 

# # def _sync_infer(tensor: torch.Tensor) -> None:
# #     with torch.inference_mode():
# #         _ = MODEL(tensor)
# #         if torch.cuda.is_available():
# #             torch.cuda.synchronize()
# #     return


# def _sync_infer(payload : torch.Tensor, model_key: str) -> None:
#     cfg = MODEL_REGISTRY[model_key]
#     kind = cfg["kind"]
#     with torch.inference_mode():
#         if kind == "cls_torch":
#             _ = cfg["model"](payload)
#         elif kind == "cls_hf":
#             _ = cfg["model"](**payload)
#         elif kind == "det":
#             _ = cfg["model"].predict(source=payload, verbose=False, device=cfg["device_arg"], batch=len(payload))
#         else:
#             raise ValueError(f"Unsupported kind: {kind}")
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#     return

# # async def load_image(image_bytes: bytes) -> torch.Tensor:
# #     loop = asyncio.get_event_loop()
# #     image = await loop.run_in_executor(None, lambda b: Image.open(io.BytesIO(b)).convert("RGB"), image_bytes)
# #     tensor = await loop.run_in_executor(None, lambda img: PREPROCESS(img).unsqueeze(0).repeat(REPEATS, 1, 1, 1).to(DEVICE), image)
# #     return tensor

# async def prepare_payload(image_bytes: bytes, cfg: dict):
#     loop = asyncio.get_event_loop()
#     image = await loop.run_in_executor(None, lambda b: Image.open(io.BytesIO(b)).convert("RGB"), image_bytes)

#     if cfg["kind"] == "cls_torch":
#         def proc(img):
#             tensor = cfg["preprocess"](img).unsqueeze(0)
#             if cfg["repeat"] > 1:
#                 tensor = tensor.repeat(cfg["repeat"], 1, 1, 1)
#             return tensor.to(DEVICE)
#         return await loop.run_in_executor(None, proc, image)

#     if cfg["kind"] == "cls_hf":
#         def proc(img):
#             inputs = cfg["processor"](images=[img] * cfg["repeat"], return_tensors="pt")
#             return {k: v.to(DEVICE) for k, v in inputs.items()}
#         return await loop.run_in_executor(None, proc, image)

#     if cfg["kind"] == "det":
#         array = np.array(image)
#         return [array.copy() for _ in range(cfg["repeat"])]

#     raise ValueError(f"Unsupported kind: {cfg['kind']}")


# app = FastAPI()

# @app.get("/ping")
# async def ping():
#     return JSONResponse(content={"message": "pong"})

# @app.get("/health")
# async def health():
#     return {"status": "ok", "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")}

# @app.post("/infer")
# async def infer(
#     model: str = Query("resnet18"),
#     image_: UploadFile = File(...),
#     tim: float = Query(0.0)
# ):
#     # image_bytes = await image_.read()
#     # tensor = await load_image(image_bytes)
#     if model not in MODEL_REGISTRY:
#         return JSONResponse(content={"error": f"unsupported model: {model}"}, status_code=400)

#     cfg = MODEL_REGISTRY[model]
#     image_bytes = await image_.read()
#     payload = await prepare_payload(image_bytes, cfg)

#     loop = asyncio.get_event_loop()
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     # await loop.run_in_executor(None, _sync_infer, tensor)
#     await loop.run_in_executor(None, _sync_infer, payload, model)
#     t1 = time.perf_counter()
#     elapsed_time = (t1 - t0) * 1000  # convert to milliseconds
#     return JSONResponse(content={
#         "model": model,
#         "elapsed_time": elapsed_time,  # in milliseconds
#         "ts": time.time(),
#         "sendtime": tim
#     })



import io
import asyncio
import logging
import time
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Query

import torch
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
import numpy as np

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api_server')

# os.environ['CUDA_VISIBLE_DEVICES']="0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VIT_HF_ID = "google/vit-large-patch16-224-in21k"
YOLO_WEIGHTS_ID = "yolov10s.pt"

YOLO_DEVICE_ARG = DEVICE.index if DEVICE.type == "cuda" else "cpu"
YOLO_MODEL = YOLO(YOLO_WEIGHTS_ID).to(YOLO_DEVICE_ARG)

# REPEATS = 32  # number of times to repeat the image in the batch dimension

MODEL_REGISTRY = {
    "resnet18": {
        "kind": "cls_torch",
        "model": resnet18(weights=ResNet18_Weights.DEFAULT).to(DEVICE).eval(),
        "preprocess": ResNet18_Weights.DEFAULT.transforms(),
        "repeat": 256,
    },
    "vitlarge": {
        "kind": "cls_hf",
        "model": ViTForImageClassification.from_pretrained(VIT_HF_ID, local_files_only=True).to(DEVICE).eval(),
        "processor": ViTImageProcessor.from_pretrained(VIT_HF_ID, local_files_only=True),
        "repeat": 8,
    },
    "yolov10-s": {
        "kind": "det",
        "model": YOLO_MODEL,
        "repeat": 32,
        "device_arg": YOLO_DEVICE_ARG,
    },
}

# 281.0835163087285 / 128
# 351.35016059395025 / 256 
# 

# def _sync_infer(tensor: torch.Tensor) -> None:
#     with torch.inference_mode():
#         _ = MODEL(tensor)
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#     return


def _sync_infer(payload : torch.Tensor, model_key: str) -> None:
    cfg = MODEL_REGISTRY[model_key]
    kind = cfg["kind"]
    with torch.inference_mode():
        if kind == "cls_torch":
            _ = cfg["model"](payload)
        elif kind == "cls_hf":
            _ = cfg["model"](**payload)
        elif kind == "det":
            _ = cfg["model"].predict(source=payload, verbose=False, device=cfg["device_arg"], batch=len(payload))
        else:
            raise ValueError(f"Unsupported kind: {kind}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return

# async def load_image(image_bytes: bytes) -> torch.Tensor:
#     loop = asyncio.get_event_loop()
#     image = await loop.run_in_executor(None, lambda b: Image.open(io.BytesIO(b)).convert("RGB"), image_bytes)
#     tensor = await loop.run_in_executor(None, lambda img: PREPROCESS(img).unsqueeze(0).repeat(REPEATS, 1, 1, 1).to(DEVICE), image)
#     return tensor

async def prepare_payload(image_bytes: bytes, cfg: dict):
    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(None, lambda b: Image.open(io.BytesIO(b)).convert("RGB"), image_bytes)

    if cfg["kind"] == "cls_torch":
        def proc(img):
            tensor = cfg["preprocess"](img).unsqueeze(0)
            if cfg["repeat"] > 1:
                tensor = tensor.repeat(cfg["repeat"], 1, 1, 1)
            return tensor.to(DEVICE)
        return await loop.run_in_executor(None, proc, image)

    if cfg["kind"] == "cls_hf":
        def proc(img):
            inputs = cfg["processor"](images=[img] * cfg["repeat"], return_tensors="pt")
            return {k: v.to(DEVICE) for k, v in inputs.items()}
        return await loop.run_in_executor(None, proc, image)

    if cfg["kind"] == "det":
        array = np.array(image)
        return [array.copy() for _ in range(cfg["repeat"])]

    raise ValueError(f"Unsupported kind: {cfg['kind']}")


app = FastAPI()

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"})

@app.get("/health")
async def health():
    return {"status": "ok", "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")}

@app.post("/infer")
async def infer(
    model: str = Query("resnet18"),
    image_: UploadFile = File(...),
    tim: float = Query(0.0)
):
    # image_bytes = await image_.read()
    # tensor = await load_image(image_bytes)
    if model not in MODEL_REGISTRY:
        return JSONResponse(content={"error": f"unsupported model: {model}"}, status_code=400)

    cfg = MODEL_REGISTRY[model]
    image_bytes = await image_.read()
    payload = await prepare_payload(image_bytes, cfg)

    loop = asyncio.get_event_loop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # await loop.run_in_executor(None, _sync_infer, tensor)
    await loop.run_in_executor(None, _sync_infer, payload, model)
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000  # convert to milliseconds
    return JSONResponse(content={
        "model": model,
        "elapsed_time": elapsed_time,  # in milliseconds
        "ts": time.time(),
        "sendtime": tim
    })
