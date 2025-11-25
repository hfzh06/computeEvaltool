import time
import random
import asyncio
from pathlib import Path
import logging
from typing import Tuple, List, Dict

import httpx
import numpy as np

logger = logging.getLogger('ComputeEvaltool.vision_benchmark')


async def preload_image(image_folder: Path, image_count: int = 100) -> list[tuple[str, bytes]]:
    """预加载多张图片"""
    image_files = list(image_folder.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No .jpg files found in {image_folder}")
    chosen = random.sample(image_files, k=min(image_count, len(image_files)))
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, file.read_bytes) for file in chosen]
    contents = await asyncio.gather(*tasks)
    logger.info(f"Preloaded {len(contents)} images from {image_folder}")
    return [(file.name, content) for file, content in zip(chosen, contents)]


async def worker(
    client: httpx.AsyncClient,
    url: str,
    images: list[tuple[str, bytes]],
    model_name: str,
    timeout: float,
    worker_id: int = 0
) -> Tuple[int, List[float], List[float]]:
    """单个 worker 发送请求"""
    end_time = time.time() + timeout
    count = 0
    elapseds = []
    responsetime = []
    
    while time.time() < end_time:
        filename, image = random.choice(images)
        try:
            send_time = time.time()
            response = await client.post(
                url,
                files={"image_": image},
                params={"model": model_name, "tim": send_time},
                timeout=1000000000
            )
            if response.status_code == 200:
                count += 1
                resp = response.json()
                if "elapsed_time" in resp:
                    elapseds.append(resp.get("elapsed_time"))
                if "ts" in resp:
                    responsetime.append(resp.get("ts") - resp.get("sendtime"))
        except Exception as e:
            logger.debug(f"[Worker {worker_id}] Request failed: {e}")
            continue
    
    logger.debug(f"[Worker {worker_id}] Completed {count} requests")
    return count, elapseds, responsetime


async def benchmark_single(
    urls: List[str],
    image_folder: Path,
    model_name: str,
    concurrency: int,
    timeout: int,
    image_count: int = 100
) -> Tuple[List[float], List[float], int]:
    """对单个模型+并发数组合进行 benchmark"""
    # 预加载图片（所有 endpoint 共享）
    images = await preload_image(image_folder, image_count)
    
    limits = httpx.Limits(
        max_connections=1024,
        max_keepalive_connections=1024,
        keepalive_expiry=100000
    )
    
    # 为每个 URL 创建并发 worker
    all_tasks = []
    async with httpx.AsyncClient(limits=limits) as client:
        worker_id = 0
        for url in urls:
            for _ in range(concurrency):
                task = worker(client, url, images, model_name, timeout, worker_id)
                all_tasks.append(task)
                worker_id += 1
        
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # 汇总结果
    total_requests = 0
    total_elapseds = []
    total_responsetime = []
    
    for r in results:
        if isinstance(r, Exception):
            logger.exception("Worker failed", exc_info=r)
            continue
        
        total_requests += r[0]
        total_elapseds.extend(r[1])
        total_responsetime.extend(r[2])
    
    if total_elapseds:
        avg = sum(total_elapseds) / len(total_elapseds)
    else:
        avg = float("nan")
    
    logger.info(f"Model: {model_name}, Conc: {concurrency}, Total requests: {total_requests}, Avg latency: {avg:.3f}ms")
    
    return total_elapseds, total_responsetime, total_requests


async def run_benchmark(args) -> Dict[str, Dict[int, Tuple[List[float], List[float], int]]]:
    """运行完整的 benchmark 测试
    
    返回格式: {model_name: {concurrency: (elapseds, responsetimes, total_requests)}}
    """
    # 生成URL列表
    if hasattr(args, 'port_range') and args.port_range:
        start, end = map(int, args.port_range.split('-'))
        ports = list(range(start, end + 1))
    else:
        ports = args.ports
    
    urls = [
        f"http://{host}:{port}/infer"
        for host in args.hosts
        for port in ports
    ]
    
    logger.info(f"Testing {len(urls)} endpoints")
    logger.info(f"Models: {args.models}")
    logger.info(f"Concurrency levels: {args.concurrency}")
    logger.info(f"Duration: {args.timeout}s per test")
    
    # 结果字典
    results = {}
    
    # 遍历所有模型和并发数组合
    for model in args.models:
        results[model] = {}
        for conc in args.concurrency:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Model: {model}, Concurrency: {conc}")
            logger.info(f"{'='*60}")
            
            result = await benchmark_single(
                urls,
                args.image_folder,
                model,
                conc,
                args.timeout,
                getattr(args, 'image_count', 100)
            )
            results[model][conc] = result
    
    return results


def calculate_percentiles(data: List[float], percentiles: List[int] = [50, 90, 95, 99]) -> dict:
    """计算百分位数"""
    if not data:
        return {f'p{p}': float('nan') for p in percentiles}
    return {f'p{p}': np.percentile(data, p) for p in percentiles}