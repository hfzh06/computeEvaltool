import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import ray
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
import random

# --- 配置区 ---
IMAGE_PATH = "/root/cocodataset/val2017"
MODELS_TO_TEST = ["resnet-18", "yolov10-s", "vit-large"]

VIT_HF_ID = "google/vit-large-patch16-224-in21k"
YOLO_WEIGHTS_ID = "yolov10s.pt"
TEST_DURATION_SECONDS = 10 # 测试时长

# ✅ 核心修改：每个模型对应固定的 batch_size 和不同的 CLIENT_CONCURRENCY
MODEL_CONFIGS = {
    "resnet-18": {
        "batch_size": 256,
        "client_concurrency_list": [64, 90, 128]  # 可以调整这个列表
    },
    "yolov10-s": {
        "batch_size": 32,
        "client_concurrency_list": [64, 90, 128]
    },
    "vit-large": {
        "batch_size": 8,
        "client_concurrency_list": [64, 90, 128]
    }
}

def load_images_from_dir(dir_path):
    """从指定目录加载所有支持的图片文件"""
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.isdir(dir_path):
        print(f"⚠️ 目录 {dir_path} 不存在，生成随机噪音图用于测试...")
        return [Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)) for _ in range(10)]

    image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                   if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if len(image_paths) > 200: 
        image_paths = image_paths[:200]

    print(f"✅ 从 '{dir_path}' 目录加载了 {len(image_paths)} 张图片。")
    return [Image.open(p).convert("RGB") for p in image_paths]

@ray.remote(num_gpus=1)
class BenchmarkActor:
    def __init__(self, model_name, vit_hf_id, yolo_weights_id):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.kind = ""

        print(f"🚀 Actor initializing on GPU {torch.cuda.current_device()} | Model: {model_name}...")

        if model_name == "resnet-18":
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            self.processor = weights.transforms()
            self.kind = "cls_torch"
        elif model_name == "vit-large":
            self.model = ViTForImageClassification.from_pretrained(vit_hf_id)
            self.processor = ViTImageProcessor.from_pretrained(vit_hf_id)
            self.kind = "cls_hf"
        elif model_name == "yolov10-s":
            self.model = YOLO(yolo_weights_id)
            self.processor = None
            self.kind = "det"
        
        if self.kind != "det":
            self.model = self.model.to(self.device)
            self.model.eval()

        print(f"🔥 Actor on GPU {torch.cuda.current_device()} warming up...")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        self.run_benchmark(1, dummy_img)
        print(f"✅ Actor on GPU {torch.cuda.current_device()} ready.")

    def run_benchmark(self, batch_size, image_array, internal_iters=10):
        image = Image.fromarray(image_array)
        
        if self.kind == "cls_torch":
            tensor_one = self.processor(image)
            tensor_one = tensor_one.unsqueeze(0)
            x = tensor_one.repeat(batch_size, 1, 1, 1).to(self.device)

        elif self.kind == "cls_hf":
            inputs_one = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs_one['pixel_values'].repeat(batch_size, 1, 1, 1)
            inputs = {'pixel_values': pixel_values.to(self.device)}

        elif self.kind == "det":
            imgs = [image] * batch_size

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            if self.kind == "cls_torch":
                _ = self.model(x)
            elif self.kind == "cls_hf":
                _ = self.model(**inputs).logits
            elif self.kind == "det":
                _ = self.model.predict(source=imgs, verbose=False, device=self.device, batch=batch_size)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000
        total_images = batch_size * internal_iters
        
        return {
            "latency_ms": latency_ms,
            "batch_size": batch_size,
            "total_images_processed": total_images,
            "gpu_id": torch.cuda.current_device()
        }

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto", ignore_reinit_error=True)

    images = load_images_from_dir(IMAGE_PATH)
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"⚡ 检测到 Ray 集群共有 {n_gpus} 张 GPU")

    if n_gpus == 0:
        print("❌ 错误：集群中没有 GPU，无法运行 Benchmark。")
        exit()

    for model in MODELS_TO_TEST:
        print(f"\n{'='*60}\n🤖 模型: {model}\n{'='*60}")

        # 获取该模型的配置
        config = MODEL_CONFIGS[model]
        batch_size = config["batch_size"]
        concurrency_list = config["client_concurrency_list"]

        # 创建 Actors
        actors = [BenchmarkActor.remote(model, VIT_HF_ID, YOLO_WEIGHTS_ID) for _ in range(n_gpus)]
        
        print("等待 Actors 初始化及预热...")
        time.sleep(5) 

        # ✅ 修改：遍历不同的 CLIENT_CONCURRENCY
        for client_concurrency in concurrency_list:
            print(f"\n📊 配置: Batch Size = {batch_size}, Client Concurrency = {client_concurrency}")
            print(f"⏱️ 测试时长: {TEST_DURATION_SECONDS} 秒")

            stats = {
                "server_latencies": [],
                "e2e_latencies": [],
                "total_images": 0,
                "total_requests": 0
            }
            
            future_to_actor_idx = {} 
            futures_in_flight = []
            submit_time_map = {}

            start_time = time.perf_counter()

            # --- 1. 填满初始请求池 ---
            for i in range(client_concurrency):  # ✅ 使用当前的 client_concurrency
                actor_idx = i % n_gpus
                actor = actors[actor_idx]
                
                img_array = np.array(random.choice(images))
                
                submit_ts = time.perf_counter()
                fut = actor.run_benchmark.remote(batch_size, img_array)
                
                futures_in_flight.append(fut)
                future_to_actor_idx[fut] = actor_idx
                submit_time_map[fut] = submit_ts

            # --- 2. 循环处理直到时间结束 ---
            while time.perf_counter() - start_time < TEST_DURATION_SECONDS:
                done_futures, futures_in_flight = ray.wait(futures_in_flight, num_returns=1)

                if not done_futures:
                    continue

                for fut in done_futures:
                    result = ray.get(fut)
                    actor_idx = future_to_actor_idx.pop(fut)
                    submit_ts = submit_time_map.pop(fut)
                    
                    now = time.perf_counter()
                    e2e_ms = (now - submit_ts) * 1000
                    
                    stats["server_latencies"].append(result["latency_ms"])
                    stats["e2e_latencies"].append(e2e_ms)
                    stats["total_requests"] += 1
                    stats["total_images"] += result["batch_size"]

                    if time.perf_counter() - start_time < TEST_DURATION_SECONDS:
                        new_actor = actors[actor_idx]
                        img_array = np.array(random.choice(images))
                        new_submit_ts = time.perf_counter()
                        
                        new_fut = new_actor.run_benchmark.remote(batch_size, img_array)
                        
                        futures_in_flight.append(new_fut)
                        future_to_actor_idx[new_fut] = actor_idx
                        submit_time_map[new_fut] = new_submit_ts

            # --- 3. 收集剩余任务 ---
            if futures_in_flight:
                remaining_results = ray.get(futures_in_flight)
                now = time.perf_counter()
                for fut, res in zip(futures_in_flight, remaining_results):
                    submit_ts = submit_time_map.pop(fut)
                    e2e_ms = (now - submit_ts) * 1000
                    stats["server_latencies"].append(res["latency_ms"])
                    stats["e2e_latencies"].append(e2e_ms)
                    stats["total_requests"] += 1
                    stats["total_images"] += res["batch_size"]

            actual_duration = time.perf_counter() - start_time

            # --- 4. 最终统计与输出 ---
            server_lats = np.array(stats["server_latencies"])
            e2e_lats = np.array(stats["e2e_latencies"])
            
            tps = stats["total_images"] / actual_duration
            rps = stats["total_requests"] / actual_duration

            print(f"\n🏆 {model} Results (Batch={batch_size}, Concurrency={client_concurrency}):")
            print(f"  - Total Images Processed: {stats['total_images']}")
            print(f"  - Actual Duration: {actual_duration:.2f} s")
            print(f"  - Throughput (TPS): {tps:.2f} images/s")
            print(f"  - QPS: {rps:.2f} requests/s")
            print("-" * 30)
            if len(server_lats) > 0:
                print(f"  - Server Latency (Model only):")
                print(f"    Avg: {np.mean(server_lats):.2f} ms")
                print(f"    P50: {np.percentile(server_lats, 50):.2f} ms")
                print(f"    P95: {np.percentile(server_lats, 95):.2f} ms")
                print(f"    P99: {np.percentile(server_lats, 99):.2f} ms")
                print(f"  - Client E2E Latency (Include Queue+Network):")
                print(f"    Avg: {np.mean(e2e_lats):.2f} ms")
                print(f"    P95: {np.percentile(e2e_lats, 95):.2f} ms")
            print("="*60)

    ray.shutdown()