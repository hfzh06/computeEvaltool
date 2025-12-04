import asyncio
import logging
from pathlib import Path

from computeEvaltool.utils.log import get_logger
from .benchmark import run_benchmark
from .report import print_benchmark_summary

logger = get_logger()


def run_vision_benchmark(args):
    """运行 vision model benchmark 的主函数"""
    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 关闭HTTP库的日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # 验证 image folder
    if not args.image_folder.exists():
        logger.error(f"Image folder not found: {args.image_folder}")
        return
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Starting Vision Model Benchmark")
    logger.info("="*60)
    logger.info(f"Models: {args.models}")
    logger.info(f"Concurrency levels: {args.concurrency}")
    logger.info(f"Duration per test: {args.timeout}s")
    logger.info(f"Total tests: {len(args.models) * len(args.concurrency)}")
    logger.info(f"Hosts: {args.hosts}")
    logger.info(f"Ports: {args.ports}")
    logger.info(f"Image folder: {args.image_folder}")
    logger.info("="*80)
    
    # 运行 benchmark
    results = asyncio.run(run_benchmark(args))
    
    # 打印和保存结果
    output_dir = None if args.no_excel else args.output_dir
    print_benchmark_summary(
        results,
        args.timeout,
        output_dir
    )
    
    logger.info("="*80)
    logger.info("Benchmark completed!")
    logger.info("="*80)