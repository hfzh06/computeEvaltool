import time
import random
import asyncio
from pathlib import Path
import rich, logging
from typing import Tuple, List, Dict

import httpx
import numpy as np

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel


logger = logging.getLogger('ComputeEvaltool.vision_benchmark')
console = Console()

MODEL_BATCH_SIZES = {
    "resnet18": 256,
    "yolov10-s": 32,
    "vitlarge": 8,
}

def _setup_rich_logger():
    """确保 logger 使用 RichHandler 输出彩色日志。"""
    if any(isinstance(h, RichHandler) for h in logger.handlers):
        return
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_setup_rich_logger()

def _print_section(title: str, style: str = "cyan"):
    """输出带颜色的分隔标题。"""
    console.rule(f"[bold {style}]{title}[/bold {style}]")

def _print_panel(message: str, title: str = "INFO", style: str = "green"):
    """显示关键信息面板。"""
    console.print(Panel.fit(message, title=f"[bold]{title}[/bold]", border_style=style))

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
                receive_time = time.time()
                if "elapsed_time" in resp:
                    elapseds.append(resp.get("elapsed_time"))
                if "ts" in resp:
                    # responsetime.append(resp.get("ts") - resp.get("sendtime"))
                    responsetime.append((receive_time - send_time))
        except Exception as e:
            logger.debug(f"[Worker {worker_id}] Request failed: {e}")
            continue
    
    logger.debug(f"[Worker {worker_id}] Completed {count} requests")
    return count, elapseds, responsetime

async def run_benchmark_adaptive(
    args,
    max_concurrency: int = 100,
    rps_threshold: float = 0.05,
    latency_threshold: float = 0.1
) -> Dict[str, Dict[int, Tuple[List[float], List[float], int]]]:
    """自适应并发数的benchmark测试
    
    Args:
        args: 命令行参数
        max_concurrency: 最大并发数限制
        rps_threshold: RPS增长阈值 默认5%
        latency_threshold: 延迟增长阈值 默认10%
    
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
    
    num_endpoints = len(urls)
    # logger.info(f"Testing {len(urls)} endpoints")
    # logger.info(f"Models: {args.models}")
    # logger.info(f"Adaptive concurrency testing (max: {max_concurrency})")
    # logger.info(f"RPS increase threshold: {rps_threshold * 100}%")
    # logger.info(f"Duration: {args.timeout}s per test")
    logger.info(f"[cyan]Testing {len(urls)} endpoints[/]  |  "
                f"[cyan]Models[/]: {', '.join(args.models)}  |  "
                f"[cyan]Duration[/]: {args.timeout}s per test")
    logger.info(f"[cyan]Adaptive max concurrency[/]: {max_concurrency}  |  "
                f"[cyan]RPS threshold[/]: {rps_threshold*100:.1f}%  |  "
                f"[cyan]Latency threshold[/]: {latency_threshold*100:.1f}%")    
    
    results = {}
    
    for model in args.models:
        # logger.info(f"\n{'='*60}")
        # logger.info(f"Testing Model: {model}")
        # logger.info(f"{'='*60}")
        _print_section(f"Model • {model}", "magenta")
        logger.info(f"[bold white]Starting saturation test for[/] [bold cyan]{model}[/]")
        
        results[model] = {}
        concurrency = 2
        rps_history = []  # 记录最近的RPS值
        latency_history = []  # 记录最近的延迟值
        
        while concurrency <= max_concurrency:
            effective_concurrency = concurrency * num_endpoints * MODEL_BATCH_SIZES.get(model, 1)
            # logger.info(f"\n--- Testing Concurrency: {concurrency} ---")
            logger.info(f"[yellow]→ Testing concurrency[/] [bold]{effective_concurrency}[/bold]"
                        f"(per-endpoint {concurrency * MODEL_BATCH_SIZES.get(model, 1)})")         
            
            result = await benchmark_single(
                urls,
                args.image_folder,
                model,
                concurrency,
                args.timeout,
                getattr(args, 'image_count', 100)
            )
            
            elapseds, responsetimes, total_requests = result
            current_rps = total_requests / args.timeout if args.timeout > 0 else 0
            avg_latency = (sum(elapseds) / len(elapseds)) if elapseds else float('inf')
            
            results[model][effective_concurrency] = result
            rps_history.append(current_rps)
            latency_history.append(avg_latency)
            
            # logger.info(f"Concurrency {concurrency}: RPS = {current_rps:.2f}, Avg latency = {avg_latency if avg_latency is not None else float('nan'):.3f} ms")
            logger.info(f"[green]✔ Concurrency {effective_concurrency}[/]: "
                        f"RPS = [bold]{current_rps * MODEL_BATCH_SIZES.get(model, 1):.2f}[/], "
                        f"Avg latency = [bold]{avg_latency / MODEL_BATCH_SIZES.get(model, 1) :.3f} ms[/]")           
            # 检查是否需要停止测试
            if len(rps_history) >= 3:
                # 计算最近两次的RPS增长率
                rps_growth_1 = (rps_history[-1] - rps_history[-2]) / rps_history[-2] if rps_history[-2] > 0 else float('inf')
                rps_growth_2 = (rps_history[-2] - rps_history[-3]) / rps_history[-3] if rps_history[-3] > 0 else float('inf')
                
                logger.info(f"RPS growth: prev={rps_growth_1:.2%}, prev-prev={rps_growth_2:.2%}")
                
                latency_growth_1 = (latency_history[-1] - latency_history[-2]) / latency_history[-2] if latency_history[-2] > 0 else float('inf')
                latency_growth_2 = (latency_history[-2] - latency_history[-3]) / latency_history[-3] if latency_history[-3] > 0 else float('inf')

                logger.info(f"Latency growth: prev={latency_growth_1:.2%}, prev-prev={latency_growth_2:.2%}")


                if (
                    rps_growth_1 < rps_threshold and
                    rps_growth_2 < rps_threshold and
                    latency_growth_1 > latency_threshold and
                    latency_growth_2 > latency_threshold
                ):
                    # logger.info(f"✓ Stopping test: RPS growth below {rps_threshold * 100}% and Latency growth above {latency_threshold * 100}% for 2 consecutive tests")
                    # logger.info(f"Optimal concurrency for {model}: {concurrency}")
                    logger.info(
                        f"[bold green]✓ Stop condition met[/]: "
                        f"ΔRPS<{rps_threshold*100:.1f}% & ΔLatency>{latency_threshold*100:.1f}% (2×)"
                    )
                    _print_panel(
                        message=f"Optimal concurrency for [bold]{model}[/bold]: [bold cyan]{effective_concurrency}[/bold cyan]",
                        title="Saturation Result"
                    )
                    break
                # # 连续两次增长都小于阈值，停止测试
                # if rps_growth_1 < rps_threshold and rps_growth_2 < rps_threshold:
                #     logger.info(f"✓ Stopping test: RPS growth below {rps_threshold * 100}% for 2 consecutive tests")
                #     logger.info(f"Optimal concurrency for {model}: {concurrency}")
                #     break
            
            concurrency += 1
        
        if concurrency > max_concurrency:
            # logger.warning(f"⚠ Reached maximum concurrency ({max_concurrency}) for {model}")
            logger.warning(f"[yellow]⚠ Reached maximum concurrency ({max_concurrency}) for[/] {model}")
    
    return results


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
    
    effective_concurrency = concurrency * len(urls) * MODEL_BATCH_SIZES.get(model_name, 1)
    # logger.info(f"Model: {model_name}, Conc: {concurrency}, Total requests: {total_requests}, Avg latency: {avg:.3f}ms")
    logger.info(
        f"Model: {model_name}, Conc: {effective_concurrency} (per-endpoint {concurrency * MODEL_BATCH_SIZES.get(model_name, 1)}), "
        f"Total requests: {total_requests}, Avg latency per req: {avg:.3f}ms"
    )
    return total_elapseds, total_responsetime, total_requests


async def run_benchmark(args) -> Dict[str, Dict[int, Tuple[List[float], List[float], int]]]:
    """运行完整的 benchmark 测试
    
    返回格式: {model_name: {concurrency: (elapseds, responsetimes, total_requests)}}
    """

    if(hasattr(args, 'adaptive') and args.adaptive):
        return await run_benchmark_adaptive(
            args,
            max_concurrency=getattr(args, 'max_concurrency', 100),
            rps_threshold=getattr(args, 'rps_threshold', 0.05),
            latency_threshold=getattr(args, 'latency_threshold', 0.1)
        )
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

    num_endpoints = len(urls)
    
    # 遍历所有模型和并发数组合
    for model in args.models:
        results[model] = {}
        for conc in args.concurrency:
            effective_concurrency = conc * num_endpoints * MODEL_BATCH_SIZES.get(model, 1)           
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
            results[model][effective_concurrency] = result
    
    return results


def calculate_percentiles(data: List[float], percentiles: List[int] = [50, 90, 95, 99]) -> dict:
    """计算百分位数"""
    if not data:
        return {f'p{p}': float('nan') for p in percentiles}
    return {f'p{p}': np.percentile(data, p) for p in percentiles}