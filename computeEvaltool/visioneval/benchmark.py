import time
import random
import asyncio
from pathlib import Path
import logging
from typing import Tuple, List

import httpx
import numpy as np

logger = logging.getLogger('ComputeEvaltool vision_benchmark')


async def preload_image(image_folder: Path, image_count: int = 100) -> list[tuple[str, bytes]]:
    image_files = list(image_folder.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No .jpg files found in {image_folder}")
    chosen = random.sample(image_files, k=min(image_count, len(image_files)))
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, file.read_bytes) for file in chosen]
    contents = await asyncio.gather(*tasks)
    return [(str(file), content) for file, content in zip(chosen, contents)]


async def worker(
    client: httpx.AsyncClient,
    url: str,
    images: list[tuple[str, bytes]],
    model_name: str,
    timeout: float
) -> Tuple[int, List[float], List[float]]:
    end_time = time.time() + timeout
    count = 0
    elapseds = []
    responsetime = []
    while time.time() < end_time:
        filename, image = random.choice(images)
        logger.info(f"[worker] Using image: {filename}")
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
            logger.warning(f"Request failed: {e}")
            continue
    return count, elapseds, responsetime


async def benchmark_endpoint(
    url: str,
    image_folder: Path,
    model_name: str,
    concurrency: int,
    timeout: int
) -> Tuple[List[float], List[float], int]:
    """对单个endpoint进行benchmark"""
    images = await preload_image(image_folder, 1)
    
    limits = httpx.Limits(
        max_connections=1024,
        max_keepalive_connections=1024,
        keepalive_expiry=100000
    )
    
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            worker(client, url, images, model_name, timeout)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
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
    
    logger.info(f"Endpoint {url}: {total_requests} requests, avg latency: {avg:.3f}ms")
    
    return total_elapseds, total_responsetime, total_requests


async def run_benchmark(args) -> List[Tuple[List[float], List[float], int]]:
    """运行完整的benchmark测试"""
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
    
    logger.info(f"Testing {len(urls)} endpoints with model: {args.model}")
    logger.info(f"Concurrency: {args.concurrency}, Duration: {args.timeout}s")
    
    # 创建所有benchmark任务
    tasks = []
    for url in urls:
        task = asyncio.create_task(
            benchmark_endpoint(
                url,
                args.image_folder,
                args.model,
                args.concurrency,
                args.timeout
            )
        )
        tasks.append(task)
    
    # 并发执行所有测试
    results = await asyncio.gather(*tasks)
    
    return results


def calculate_percentiles(data: List[float], percentiles: List[int] = [50, 90, 95, 99]) -> dict:
    """计算百分位数"""
    if not data:
        return {f'p{p}': float('nan') for p in percentiles}
    return {f'p{p}': np.percentile(data, p) for p in percentiles}