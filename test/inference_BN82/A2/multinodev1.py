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
import math
from pathlib import Path

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table


rand_id = int(time.time() - 1763905562) + random.randint(1, 9999)
REPORT_XLSX_PATH = Path(f"./results/vision_benchmark_multi_model_{rand_id}.xlsx")
REPORT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("multinode.benchmark")
console = Console()

def _setup_rich_logger():
    if any(isinstance(h, RichHandler) for h in logger.handlers):
        return
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

def _print_section(title: str, style: str = "cyan"):
    console.rule(f"[bold {style}]{title}[/bold {style}]")

def _print_panel(message: str, title: str = "INFO", style: str = "green"):
    console.print(Panel.fit(message, title=f"[bold]{title}[/bold]", border_style=style))

_setup_rich_logger()


# --- 配置区 ---
IMAGE_PATH = "/root/cocodataset/val2017"
MODELS_TO_TEST = ["resnet-18", "yolov10-s","vit-large"]

VIT_HF_ID = "google/vit-large-patch16-224-in21k"
YOLO_WEIGHTS_ID = "yolov10s.pt"
TEST_DURATION_SECONDS = 10 # 测试时长

# ✅ 核心修改：每个模型对应固定的 batch_size 和不同的 CLIENT_CONCURRENCY
MODEL_CONFIGS = {
    "resnet-18": {
        "batch_size": 256,
        "initial_client_concurrency": 16,
        "concurrency_increment": 16,
        "tps_gain_threshold": 0.05,
        "e2e_gain_threshold": 0.10
    },
    "yolov10-s": {
        "batch_size": 32,
        "initial_client_concurrency": 16,
        "concurrency_increment": 8,
        "tps_gain_threshold": 0.05,
        "e2e_gain_threshold": 0.10
    },
    "vit-large": {
        "batch_size": 8,
        "initial_client_concurrency": 16,
        "concurrency_increment": 8,
        "tps_gain_threshold": 0.05,
        "e2e_gain_threshold": 0.10
    }
}

def load_images_from_dir(dir_path):
    """从指定目录加载所有支持的图片文件"""
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.isdir(dir_path):
        logger.warning(f"[yellow]目录 {dir_path} 不存在，生成随机噪声图用于测试[/]")
        return [Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)) for _ in range(10)]

    image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                   if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if len(image_paths) > 200: 
        image_paths = image_paths[:200]

    # print(f"✅ 从 '{dir_path}' 目录加载了 {len(image_paths)} 张图片。")
    logger.info(f"[green]Loaded {len(image_paths)} images from[/] {dir_path}")
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
            self.model = ViTForImageClassification.from_pretrained(vit_hf_id, local_files_only=True)
            self.processor = ViTImageProcessor.from_pretrained(vit_hf_id, local_files_only=True)
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

def _render_summary_table(rows):
    if not rows:
        logger.warning("暂无可展示的结果。")
        return
    table = Table(
        title="📈 Multi-Node Benchmark Snapshot",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Model", justify="left")
    table.add_column("Conc.", justify="right")
    table.add_column("Avg Lat./img (ms)", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("RPS", justify="right")

    for row in rows:
        table.add_row(
            row["model"],
            str(row["total_concurrency"]),
            "-" if math.isnan(row["avg_lat_per_img_ms"]) else f"{row['avg_lat_per_img_ms']:.2f}",
            f"{row['qps']:.2f}",
            f"{row['rps']:.2f}",
        )
    console.print(table)

def _export_results_to_excel(rows, output_path: Path):
    if not rows:
        logger.warning("没有结果可写入 Excel。")
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df = df[
            [
                "model",
                "batch_size",
                "client_concurrency",
                "total_concurrency",
                "avg_lat_per_img_ms",
                "qps",
                "rps",
                "total_requests",
                "total_images",
                "duration",
            ]
        ]
        df.columns = [
            "Model",
            "Batch Size",
            "Client Concurrency",
            "Total Concurrency",
            "Avg Lat./img (ms)",
            "QPS",
            "RPS",
            "Total Requests",
            "Total Images",
            "Duration (s)",
        ]
        df.to_excel(output_path, index=False)
        logger.info(f"📁 结果已写入 {output_path.resolve()}")
    except Exception as exc:
        logger.exception(f"写入 Excel 失败: {exc}")


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="auto", ignore_reinit_error=True)

    images = load_images_from_dir(IMAGE_PATH)
    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    # print(f"⚡ 检测到 Ray 集群共有 {n_gpus} 张 GPU")
    logger.info(f"[cyan]Ray GPUs detected[/]: {n_gpus}")

    if n_gpus == 0:
        print("❌ 错误：集群中没有 GPU，无法运行 Benchmark。")
        exit()

    result_rows = []
    for model in MODELS_TO_TEST:
        _print_section(f"模型 • {model}", "magenta")
        config = MODEL_CONFIGS[model]
        # print(f"\n{'='*60}\n🤖 模型: {model}\n{'='*60}")

        config = MODEL_CONFIGS[model]
        batch_size = config["batch_size"]
        client_concurrency = config.get("initial_client_concurrency", 32)
        concurrency_increment = config.get("concurrency_increment", 16)
        tps_gain_threshold = config.get("tps_gain_threshold", 0.05) * 100
        e2e_gain_threshold = config.get("e2e_gain_threshold", 0.10) * 100

        actors = [BenchmarkActor.remote(model, VIT_HF_ID, YOLO_WEIGHTS_ID) for _ in range(n_gpus)]
        logger.info(
            f"[bold white]启动 {model}[/bold white] | Batch={batch_size}, "
            f"Init concurrency={client_concurrency}, ΔRPS阈值={tps_gain_threshold:.1f}%, "
            f"ΔAvg Lat.阈值={e2e_gain_threshold:.1f}%"
        )
        # print("等待 Actors 初始化及预热...")
        logger.info("[cyan]等待 Actors 初始化及预热...[/cyan]")
        time.sleep(5)

        prev_tps = None
        prev_e2e_avg = None
        low_tps_streak = 0

        while True:
            logger.info(
                f"[yellow]→ 测试配置[/]: Batch={batch_size}, Clients={client_concurrency}, "
                f"总并发={batch_size * client_concurrency}"
            )
            # print(f"\n📊 配置: Batch Size = {batch_size}, Concurrency Per Client = {client_concurrency}, Conc. = {batch_size * client_concurrency}")
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

            for i in range(client_concurrency):
                actor_idx = i % n_gpus
                actor = actors[actor_idx]

                img_array = np.array(random.choice(images))

                submit_ts = time.perf_counter()
                fut = actor.run_benchmark.remote(batch_size, img_array)

                futures_in_flight.append(fut)
                future_to_actor_idx[fut] = actor_idx
                submit_time_map[fut] = submit_ts

            while time.perf_counter() - start_time < TEST_DURATION_SECONDS:
                done_futures, futures_in_flight = ray.wait(futures_in_flight, num_returns=1)

                if not done_futures:
                    logger.debug(f"等待返回中，当前在途 {len(futures_in_flight)} 个任务")
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

            if futures_in_flight:
                remaining_results = ray.get(futures_in_flight)
                now = time.perf_counter()
                for fut, res in zip(futures_in_flight, remaining_results):
                    future_to_actor_idx.pop(fut, None)
                    submit_ts = submit_time_map.pop(fut)
                    e2e_ms = (now - submit_ts) * 1000
                    stats["server_latencies"].append(res["latency_ms"])
                    stats["e2e_latencies"].append(e2e_ms)
                    stats["total_requests"] += 1
                    stats["total_images"] += res["batch_size"]

            actual_duration = time.perf_counter() - start_time

            server_lats = np.array(stats["server_latencies"])
            e2e_lats = np.array(stats["e2e_latencies"])

            tps = stats["total_images"] / actual_duration if actual_duration > 0 else 0.0
            rps = stats["total_requests"] / actual_duration if actual_duration > 0 else 0.0
            e2e_avg = float(np.mean(e2e_lats)) if len(e2e_lats) else 0.0
            

            avg_lat_img = (e2e_avg / batch_size) if batch_size else float("nan")
            summary_row = {
                "model": model,
                "batch_size": batch_size,
                "client_concurrency": client_concurrency,
                "total_concurrency": batch_size * client_concurrency,
                "avg_lat_per_img_ms": avg_lat_img if not math.isnan(avg_lat_img) else float("nan"),
                "qps": rps,
                "rps": tps,
                "total_requests": stats["total_requests"],
                "total_images": stats["total_images"],
                "duration": actual_duration,
            }
            result_rows.append(summary_row)
            # print(f"\n🏆 {model} Results (Batch={batch_size}, Concurrency={client_concurrency}):")
            # print(f"  - Total Images Processed: {stats['total_images']}")
            # print(f"  - Actual Duration: {actual_duration:.2f} s")
            # print(f"  - Throughput (RPS): {tps:.2f} images/s")
            # print(f"  - QPS: {rps:.2f} requests/s")
            # print("-" * 30)
            logger.info(
                f"[bold green]🏆 {model}[/] Batch={batch_size}, Clients={client_concurrency} | "
                f"Images={stats['total_images']} | Duration={actual_duration:.2f}s | "
                f"QPS={rps:.2f} | RPS={tps:.2f}"
            )
            if len(server_lats) > 0:
            #     print(f"  - Server Latency (Model only):")
            #     print(f"    Avg: {np.mean(server_lats):.2f} ms")
            #     print(f"    P50: {np.percentile(server_lats, 50):.2f} ms")
            #     print(f"    P95: {np.percentile(server_lats, 95):.2f} ms")
            #     print(f"    P99: {np.percentile(server_lats, 99):.2f} ms")
            #     print(f"  - Client E2E Latency (Include Queue+Network):")
            #     print(f"    Avg per batch: {e2e_avg:.2f} ms")
            #     print(f"    P95 per batch: {np.percentile(e2e_lats, 95):.2f} ms")
            #     print(f"    Avg Lat. : {e2e_avg / batch_size:.2f} ms")
            #     print(f"    P95 per batch: {np.percentile(e2e_lats / batch_size, 95):.2f} ms")
            # print("=" * 60)
                logger.info(
                    f"[blue]Server Latency[/]: Avg={np.mean(server_lats):.2f}ms | "
                    f"P50={np.percentile(server_lats,50):.2f} | "
                    f"P95={np.percentile(server_lats,95):.2f} | "
                    f"P99={np.percentile(server_lats,99):.2f}"
                )
                logger.info(
                    f"[blue]Client E2E[/]: Avg(batch)={e2e_avg:.2f}ms | "
                    f"P95(batch)={np.percentile(e2e_lats,95):.2f}ms | "
                    f"Avg Lat./img={e2e_avg/batch_size:.2f}ms"
                )                

            tps_gain_pct = None
            e2e_gain_pct = None

            if prev_tps is not None and prev_tps > 0:
                tps_gain_pct = ((tps - prev_tps) / prev_tps) * 100

            if prev_e2e_avg is not None and prev_e2e_avg > 0:
                raw_e2e_gain = ((e2e_avg - prev_e2e_avg) / prev_e2e_avg) * 100
                e2e_gain_pct = max(raw_e2e_gain, 0.0)

            if tps_gain_pct is not None and e2e_gain_pct is not None:
                logger.info(
                    f"ΔRPS={tps_gain_pct:.2f}% (阈值 {tps_gain_threshold:.2f}%) | "
                    f"ΔAvg Lat.={e2e_gain_pct if e2e_gain_pct is not None else 0:.2f}% "
                    f"(阈值 {e2e_gain_threshold:.2f}%)"
                )

            stop_due_to_plateau = False
            if tps_gain_pct is not None and e2e_gain_pct is not None:
                perf_degraded = (
                    tps_gain_pct <= tps_gain_threshold and
                    e2e_gain_pct >= e2e_gain_threshold
                )

                if perf_degraded:
                    low_tps_streak += 1
                else:
                    low_tps_streak = 0

                if low_tps_streak >= 2:
                    stop_due_to_plateau = True

            prev_tps = tps
            prev_e2e_avg = e2e_avg

            if stop_due_to_plateau:
                # print("连续续两轮出现 TPS 增幅≤阈值且 E2E 时延增幅≥阈值，停止该模型测试。")
                _print_panel(
                    message=f"{model} 提前停止：两轮 ΔRPS≤{tps_gain_threshold:.1f}% 且 ΔAvg Lat.≥{e2e_gain_threshold:.1f}%",
                    title="Stop Condition",
                    style="red"
                )
                break

            client_concurrency += concurrency_increment
            # print(f"➡️ 并发增加至 {client_concurrency}，继续下一轮。")
            logger.info(f"[cyan]并发提升至 {client_concurrency * batch_size}，继续测试[/cyan]")
    _print_section("综合结果", "green")
    _render_summary_table(result_rows)
    _export_results_to_excel(result_rows, REPORT_XLSX_PATH)
    ray.shutdown()


