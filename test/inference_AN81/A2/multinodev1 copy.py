# import os
# import time
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# import ray
# from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
# from transformers import ViTImageProcessor, ViTForImageClassification
# from ultralytics import YOLO
# import random

# # --- 配置区 ---
# IMAGE_PATH = "/root/cocodataset/val2017"
# # MODELS_TO_TEST = ["vit-large"] 
# # MODELS_TO_TEST = ["resnet-18"]
# MODELS_TO_TEST = ["resnet-18", "yolov10-s", "vit-large"]

# VIT_HF_ID = "google/vit-large-patch16-224-in21k"
# YOLO_WEIGHTS_ID = "yolov10s.pt"
# TEST_DURATION_SECONDS = 10 # 测试时长
# BATCH_SIZE_LIST = [256, 32, 8]     # 将原来的 CONCURRENCY_LIST 改名为 BATCH_SIZE_LIST 以更准确描述
# CLIENT_CONCURRENCY = 32    # 相当于你原来的 nn，表示同时有多少个请求在飞

# def load_images_from_dir(dir_path):
#     """从指定目录加载所有支持的图片文件"""
#     supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     if not os.path.isdir(dir_path):
#         # 如果本地没有文件夹，生成一些随机噪音图用于测试，避免报错
#         print(f"⚠️ 目录 {dir_path} 不存在，生成随机噪音图用于测试...")
#         return [Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)) for _ in range(10)]

#     image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
#                    if os.path.splitext(f)[1].lower() in supported_extensions]
    
#     # 限制加载数量，防止内存爆炸
#     if len(image_paths) > 200: 
#         image_paths = image_paths[:200]

#     print(f"✅ 从 '{dir_path}' 目录加载了 {len(image_paths)} 张图片。")
#     return [Image.open(p).convert("RGB") for p in image_paths]

# @ray.remote(num_gpus=1)
# class BenchmarkActor:
#     def __init__(self, model_name, vit_hf_id, yolo_weights_id):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model_name = model_name
#         self.kind = ""

#         print(f"🚀 Actor initializing on GPU {torch.cuda.current_device()} | Model: {model_name}...")

#         if model_name == "resnet-18":
#             weights = ResNet18_Weights.DEFAULT
#             self.model = resnet18(weights=weights)
#             self.processor = weights.transforms()
#             self.kind = "cls_torch"
#         elif model_name == "vit-large":
#             self.model = ViTForImageClassification.from_pretrained(vit_hf_id)
#             self.processor = ViTImageProcessor.from_pretrained(vit_hf_id)
#             self.kind = "cls_hf"
#         elif model_name == "yolov10-s":
#             self.model = YOLO(yolo_weights_id)
#             self.processor = None
#             self.kind = "det"
        
#         if self.kind != "det":
#             self.model = self.model.to(self.device)
#             self.model.eval()

#         # --- Warmup (至关重要) ---
#         # 随便造个假数据预热一下，消除首次推理的初始化延迟
#         print(f"🔥 Actor on GPU {torch.cuda.current_device()} warming up...")
#         dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
#         self.run_benchmark(1, dummy_img) # batch_size=1 warmup
#         print(f"✅ Actor on GPU {torch.cuda.current_device()} ready.")


#     def run_benchmark(self, batch_size, image_array, internal_iters=10):
#         """
#         :param batch_size: 目标 Batch Size
#         :param image_array: 单张图片的 numpy 数组
#         """
#         # 1. 恢复为 PIL 图片 (只做一次)
#         image = Image.fromarray(image_array)

#         # ------------------------------------------------------------------
#         # 核心修改区：只预处理一次，然后在 Tensor 层面进行复制 (Repeat/Expand)
#         # ------------------------------------------------------------------
        
#         # === 场景 1: PyTorch 原生 (ResNet) ===
#         if self.kind == "cls_torch":
#             # 预处理单张图片 -> [C, H, W]
#             tensor_one = self.processor(image)
#             # 增加 Batch 维度 -> [1, C, H, W]
#             tensor_one = tensor_one.unsqueeze(0)
#             # 在 GPU 上复制 -> [Batch_Size, C, H, W]
#             # 这里的 repeat 非常快，几乎不耗时
#             x = tensor_one.repeat(batch_size, 1, 1, 1).to(self.device)

#         # === 场景 2: HuggingFace (ViT) ===
#         elif self.kind == "cls_hf":
#             # ❌ 原来的写法（慢）：让 Processor 处理 32 张图
#             # inputs = self.processor(images=[image]*batch_size, return_tensors="pt")
            
#             # ✅ 新的写法（快）：只处理 1 张图
#             inputs_one = self.processor(images=image, return_tensors="pt")
            
#             # inputs_one['pixel_values'] 形状是 [1, 3, 224, 224]
#             # 我们只需要把它复制成 [Batch_Size, 3, 224, 224]
#             pixel_values = inputs_one['pixel_values'].repeat(batch_size, 1, 1, 1)
            
#             # 构造模型输入字典，并移到 GPU
#             inputs = {'pixel_values': pixel_values.to(self.device)}

#         # === 场景 3: YOLO (特殊情况) ===
#         elif self.kind == "det":
#             # YOLO 的 predict 接口比较封装，难以直接传 Tensor 进行 batch repeat
#             # 如果仅仅为了测压，这里只能传 list，或者深入 hack YOLO 内部
#             # 既然你是为了测 ViT，这里可以暂时保持原样，或者用 list 复制
#             imgs = [image] * batch_size

#         # ------------------------------------------------------------------
#         # 推理阶段 (基本不变)
#         # ------------------------------------------------------------------
#         torch.cuda.synchronize()
#         t0 = time.perf_counter()
        
#         with torch.no_grad():
#             if self.kind == "cls_torch":
#                 _ = self.model(x)
#             elif self.kind == "cls_hf":
#                 # ViT 模型
#                 _ = self.model(**inputs).logits
#             elif self.kind == "det":
#                 _ = self.model.predict(source=imgs, verbose=False, device=self.device, batch=batch_size)
        
#         torch.cuda.synchronize()
#         t1 = time.perf_counter()
        
#         latency_ms = (t1 - t0) * 1000
#         total_images = batch_size * internal_iters
        
#         return {
#             "latency_ms": latency_ms,
#             "batch_size": batch_size,
#             "total_images_processed": total_images,
#             "gpu_id": torch.cuda.current_device()
#         }

#     # def run_benchmark(self, batch_size, image_array):
#     #     """
#     #     :param batch_size: 相当于原来的 concurrency，指一次推理处理多少张图
#     #     :param image_array: 图片的 numpy 数组
#     #     """
#     #     image = Image.fromarray(image_array)

#     #     # 预处理阶段
#     #     if self.kind == "cls_torch":
#     #         # 构造 Batch
#     #         input_tensor = self.processor(image).unsqueeze(0)
#     #         if batch_size > 1:
#     #             input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)
#     #         x = input_tensor.to(self.device)
            
#     #     elif self.kind == "cls_hf":
#     #         # HuggingFace Processor 处理 batch
#     #         inputs = self.processor(images=[image]*batch_size, return_tensors="pt")
#     #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#     #     elif self.kind == "det":
#     #         imgs = [image] * batch_size

#     #     torch.cuda.synchronize()
#     #     t0 = time.perf_counter()
        
#     #     with torch.no_grad():
#     #         if self.kind == "cls_torch":
#     #             _ = self.model(x)
#     #         elif self.kind == "cls_hf":
#     #             _ = self.model(**inputs).logits
#     #         elif self.kind == "det":
#     #             # YOLO predict
#     #             _ = self.model.predict(source=imgs, verbose=False, device=self.device, batch=batch_size)
        
#     #     torch.cuda.synchronize()
#     #     t1 = time.perf_counter()
        
#     #     latency_ms = (t1 - t0) * 1000
        
#     #     return {
#     #         "latency_ms": latency_ms,
#     #         "batch_size": batch_size,
#     #         "gpu_id": torch.cuda.current_device()
#     #     }

# if __name__ == "__main__":
#     # 如果已有 ray 实例则连接，否则新建
#     if ray.is_initialized():
#         ray.shutdown()
#     ray.init(address="auto", ignore_reinit_error=True)

#     images = load_images_from_dir(IMAGE_PATH)
#     n_gpus = int(ray.cluster_resources().get("GPU", 0))
#     print(f"⚡ 检测到 Ray 集群共有 {n_gpus} 张 GPU")

#     if n_gpus == 0:
#         print("❌ 错误：集群中没有 GPU，无法运行 Benchmark。")
#         exit()

#     for model in MODELS_TO_TEST:
#         print(f"\n{'='*60}\n🤖 模型: {model}\n{'='*60}")

#         # 1. 创建 Actors
#         actors = [BenchmarkActor.remote(model, VIT_HF_ID, YOLO_WEIGHTS_ID) for _ in range(n_gpus)]
        
#         # 等待所有 Actor 初始化完成 (包含 Warmup)
#         # 我们可以调用一个简单的 ping 或者只要对象创建成功即可，这里简单等待一下
#         print("等待 Actors 初始化及预热...")
#         time.sleep(5) 

#         for batch_size in BATCH_SIZE_LIST:
#             print(f"\n📊 配置: Batch Size (每请求) = {batch_size}, Client Concurrency (飞行请求数) = {CLIENT_CONCURRENCY}")
#             print(f"⏱️ 测试时长: {TEST_DURATION_SECONDS} 秒")

#             # 状态追踪变量
#             stats = {
#                 "server_latencies": [], # 纯模型推理耗时
#                 "e2e_latencies": [],    # 客户端提交到收到的总耗时
#                 "total_images": 0,      # 处理的总图片数
#                 "total_requests": 0     # 处理的总请求数
#             }
            
#             # 核心：Future -> Actor Index 映射
#             # 这样我们才能知道哪个 Future 完成了，对应的 Actor 是谁，以便给它派新活
#             future_to_actor_idx = {} 
#             futures_in_flight = []
#             submit_time_map = {} # 记录提交时间用于计算 E2E 延迟

#             start_time = time.perf_counter()

#             # --- 1. 填满初始请求池 (Bootstrap) ---
#             for i in range(CLIENT_CONCURRENCY):
#                 actor_idx = i % n_gpus # 初始轮询分配
#                 actor = actors[actor_idx]
                
#                 img_array = np.array(random.choice(images))
                
#                 submit_ts = time.perf_counter()
#                 # 传递 batch_size
#                 fut = actor.run_benchmark.remote(batch_size, img_array)
                
#                 futures_in_flight.append(fut)
#                 future_to_actor_idx[fut] = actor_idx
#                 submit_time_map[fut] = submit_ts

#             # --- 2. 循环处理直到时间结束 ---
#             while time.perf_counter() - start_time < TEST_DURATION_SECONDS:
#                 # 等待至少一个完成
#                 done_futures, futures_in_flight = ray.wait(futures_in_flight, num_returns=1)

#                 if not done_futures:
#                     continue

#                 # 处理完成的任务
#                 for fut in done_futures:
#                     result = ray.get(fut)
#                     actor_idx = future_to_actor_idx.pop(fut)
#                     submit_ts = submit_time_map.pop(fut)
                    
#                     now = time.perf_counter()
#                     e2e_ms = (now - submit_ts) * 1000
                    
#                     # 记录数据
#                     stats["server_latencies"].append(result["latency_ms"])
#                     stats["e2e_latencies"].append(e2e_ms)
#                     stats["total_requests"] += 1
#                     stats["total_images"] += result["batch_size"]

#                     # 立即给这个刚空闲下来的 Actor 派发新任务
#                     if time.perf_counter() - start_time < TEST_DURATION_SECONDS:
#                         new_actor = actors[actor_idx] # ✅ 关键修正：复用同一个 Actor
#                         img_array = np.array(random.choice(images))
#                         new_submit_ts = time.perf_counter()
                        
#                         new_fut = new_actor.run_benchmark.remote(batch_size, img_array)
                        
#                         futures_in_flight.append(new_fut)
#                         future_to_actor_idx[new_fut] = actor_idx
#                         submit_time_map[new_fut] = new_submit_ts

#             # --- 3. 收集剩余还在飞的任务结果 (Optional) ---
#             # 如果你想计算这些剩余任务，可以用 ray.get。
#             # 严格的 Duration 测试通常忽略最后一批未完成的，或者等待它们完成。
#             # 这里选择等待它们完成以获得完整数据：
#             if futures_in_flight:
#                 remaining_results = ray.get(futures_in_flight)
#                 now = time.perf_counter()
#                 for fut, res in zip(futures_in_flight, remaining_results):
#                     submit_ts = submit_time_map.pop(fut)
#                     e2e_ms = (now - submit_ts) * 1000
#                     stats["server_latencies"].append(res["latency_ms"])
#                     stats["e2e_latencies"].append(e2e_ms)
#                     stats["total_requests"] += 1
#                     stats["total_images"] += res["batch_size"]

#             actual_duration = time.perf_counter() - start_time

#             # --- 4. 最终统计与输出 ---
#             server_lats = np.array(stats["server_latencies"])
#             e2e_lats = np.array(stats["e2e_latencies"])
            
#             tps = stats["total_images"] / actual_duration
#             rps = stats["total_requests"] / actual_duration # Requests per second

#             print(f"\n🏆 {model} Results (Batch={batch_size}):")
#             print(f"  - Total Images Processed: {stats['total_images']}")
#             print(f"  - Actual Duration: {actual_duration:.2f} s")
#             print(f"  - Throughput (TPS): {tps:.2f} images/s")
#             print(f"  - QPS: {rps:.2f} requests/s")
#             print("-" * 30)
#             if len(server_lats) > 0:
#                 print(f"  - Server Latency (Model only):")
#                 print(f"    Avg: {np.mean(server_lats):.2f} ms")
#                 print(f"    P50: {np.percentile(server_lats, 50):.2f} ms")
#                 print(f"    P95: {np.percentile(server_lats, 95):.2f} ms")
#                 print(f"    P99: {np.percentile(server_lats, 99):.2f} ms")
#                 print(f"  - Client E2E Latency (Include Queue+Network):")
#                 print(f"    Avg: {np.mean(e2e_lats):.2f} ms")
#                 print(f"    P95: {np.percentile(e2e_lats, 95):.2f} ms")
#             print("="*60)

#     ray.shutdown()


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
        "client_concurrency_list": [64, 128]  # 可以调整这个列表
    },
    "yolov10-s": {
        "batch_size": 32,
        "client_concurrency_list": [64, 128]
    },
    "vit-large": {
        "batch_size": 8,
        "client_concurrency_list": [64, 128]
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