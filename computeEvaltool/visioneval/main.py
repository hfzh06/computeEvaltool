import asyncio
import logging
from pathlib import Path

from computeEvaltool.utils.log import get_logger
from .benchmark import run_benchmark
from .report import print_benchmark_summary

logger = get_logger()


def run_vision_benchmark(args):
    """运行vision model benchmark的主函数"""
    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 关闭HTTP库的日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # 验证image folder
    if not args.image_folder.exists():
        logger.error(f"Image folder not found: {args.image_folder}")
        return
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting vision model benchmark...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Duration: {args.timeout}s")
    logger.info(f"Hosts: {args.hosts}")
    logger.info(f"Ports: {args.ports}")
    
    # 运行benchmark
    results = asyncio.run(run_benchmark(args))
    
    # 打印和保存结果
    output_dir = None if args.no_excel else args.output_dir
    print_benchmark_summary(
        results,
        args.model,
        args.concurrency,
        args.timeout,
        output_dir
    )
    
    logger.info("Benchmark completed!")