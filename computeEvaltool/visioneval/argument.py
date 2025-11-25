import argparse
from pathlib import Path


class VisionArguments:
    """Arguments for vision model benchmarking."""
    
    def __init__(self):
        self.models: list[str] = ["resnet18"]  # 改为复数
        self.concurrency: list[int] = [2]      # 改为列表
        self.timeout: int = 10
        self.hosts: list[str] = []
        self.ports: list[int] = [] 
        self.image_folder: Path = Path("../../cocodataset/val2017")
        self.output_dir: Path = Path("./results")
        self.no_excel: bool = False


def add_argument(parser: argparse.ArgumentParser):
    """Add arguments for vision benchmark."""
    
    # Model configuration - 支持多个模型
    parser.add_argument(
        '--model',
        '--models',
        type=str,
        nargs='+',
        default=['resnet18'],
        dest='models',
        help='Model names to benchmark (default: resnet18). Can specify multiple models.'
    )
    
    # Benchmark parameters - 支持多个并发数
    parser.add_argument(
        '--conc',
        '--concurrency',
        type=int,
        nargs='+',
        default=[2],
        dest='concurrency',
        help='Number of concurrent requests per endpoint (default: 2). Can specify multiple values.'
    )
    
    parser.add_argument(
        '--time',
        '--timeout',
        type=int,
        default=10,
        dest='timeout',
        help='Test duration in seconds (default: 10)'
    )
    
    # Server configuration
    parser.add_argument(
        '--hosts',
        type=str,
        nargs='+',
        default=['9.0.2.60'],
        help='List of host IPs (default: 9.0.2.60)'
    )
    
    parser.add_argument(
        '--ports',
        type=int,
        nargs='+',
        default=[30000],
        help='List of port numbers (default: 30000)'
    )
    
    parser.add_argument(
        '--port-range',
        type=str,
        help='Port range in format "start-end" (e.g., "30000-30002")'
    )
    
    # Data configuration
    parser.add_argument(
        '--image-folder',
        type=Path,
        default=Path('../../cocodataset/val2017'),
        help='Path to image folder for testing (default: ../../cocodataset/val2017)'
    )
    
    parser.add_argument(
        '--image-count',
        type=int,
        default=100,
        help='Number of images to preload for testing (default: 100)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results'),
        help='Directory to save results (default: ./results)'
    )
    
    parser.add_argument(
        '--no-excel',
        action='store_true',
        help='Do not generate Excel report'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )