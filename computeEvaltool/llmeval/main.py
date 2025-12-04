import asyncio
import copy
import os
import platform
import threading
import time
from argparse import Namespace

from computeEvaltool.llmeval.utils.local_server import start_app
from computeEvaltool.llmeval.utils.llmlog_util import init_swanlab, init_wandb
from computeEvaltool.utils.log import configure_logging, get_logger
from computeEvaltool.utils.model_utils import seed_everything
from .arguments import Arguments, parse_args
from .benchmark import benchmark
from .utils.db_util import get_output_path
from .utils.handler import add_signal_handlers
from .utils.display_util import print_summary

from .utils.benchmark_util import Metrics
logger = get_logger()


def run_one_benchmark(args: Arguments, output_path: str = None):
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]

    # Setup logger and output
    args.outputs_dir = output_path

    logger.info('Starting benchmark with args: ')
    logger.info(args)

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    if platform.system() != 'Windows':
        add_signal_handlers(loop)

    return loop.run_until_complete(benchmark(args))


def generate_parallel_sequence():
    # yield 2
    # yield 16
    # yield 64
    yield 128
    
    current  = 200
    while True:
        yield current
        current += 100


THROUGHPUT_KEY = Metrics.OUTPUT_TOKEN_THROUGHPUT
AVGLATENCY_KEY = Metrics.AVERAGE_LATENCY

def run_multi_benchmark(args: Arguments, output_path: str = None):
    results = []
    if args.auto_parallel:
        parallel_geneator = generate_parallel_sequence()
        previous_gen_token_per_s = None
        previous_avg_latency = None
        consecutive_low_improvement = 0
        consecutive_low_avg_latency_increase = 0
        IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement threshold
        LAENCY_THRESHOLD = 0.1  # 10% latency threshold
        
        run_index = 0
        while True:
            parallel = next(parallel_geneator)
            number = parallel * 2
            
            args.number = number
            args.parallel = parallel
            
            cur_output_path = os.path.join(output_path, f'parallel_{parallel}_number_{number}')
            os.makedirs(cur_output_path, exist_ok=True)
            
            logger.info(f'Starting benchmark run {run_index} with parallel={parallel}, nummber={number}')
            
            metrics_result, percentile_result = run_one_benchmark(args, output_path=cur_output_path)
            results.append((metrics_result, percentile_result))
            
            current_throughput = metrics_result.get(THROUGHPUT_KEY)
            current_latency = metrics_result.get(AVGLATENCY_KEY)
            
            if current_throughput is None or current_latency is None:
                logger.warning(
                    f'"{THROUGHPUT_KEY}" missing in metrics_result: {metrics_result} or' 
                    f'"{AVGLATENCY_KEY}" missing in metrics_result: {metrics_result}. Stopping benchmark.')
            else:
                if previous_gen_token_per_s is not None and previous_avg_latency is not None:
                    improvement_rate = (
                        current_throughput - previous_gen_token_per_s) / previous_gen_token_per_s
                    logger.info(
                        f'Improvement rate: {improvement_rate * 100:.2f}% '
                        f'(previous:{previous_gen_token_per_s}, current:{current_throughput})')
                    latency_increase = (
                        current_latency - previous_avg_latency) / previous_avg_latency
                    logger.info(
                        f'Average latency increase: {latency_increase * 100:.2f}% '
                        f'(previous:{previous_avg_latency}, current:{current_latency})')
                    if improvement_rate < IMPROVEMENT_THRESHOLD and latency_increase > LAENCY_THRESHOLD:
                        consecutive_low_improvement += 1
                        if consecutive_low_improvement >= 2:
                            logger.info(
                                f'Stopping benchmark: Gen token/s improvement below {IMPROVEMENT_THRESHOLD * 100}% '
                                f'and average latency increase above {LAENCY_THRESHOLD * 100}%.'
                                f'for 2 consecutive runs')
                            logger.info(f'Total runs completed: {run_index + 1}')
                            break
                    else:
                        consecutive_low_improvement = 0
                previous_gen_token_per_s = current_throughput
                previous_avg_latency = current_latency
            
            run_index += 1
            
            logger.info(f'Sleeping for {args.sleep_interval} seconds before the next run...')
            time.sleep(args.sleep_interval)
        
    else:

        number_list = copy.deepcopy(args.number)
        parallel_list = copy.deepcopy(args.parallel)
        for i, (number, parallel) in enumerate(zip(number_list, parallel_list)):
            args.number = number
            args.parallel = parallel
            # Set up output path for each run
            cur_output_path = os.path.join(output_path, f'parallel_{parallel}_number_{number}')
            os.makedirs(cur_output_path, exist_ok=True)
            # Start the benchmark
            metrics_result = run_one_benchmark(args, output_path=cur_output_path)
            # Save the results
            results.append(metrics_result)
            # Sleep between runs to avoid overwhelming the server
            if i < len(number_list) - 1:
                logger.info(f'Sleeping for {args.sleep_interval} seconds before the next run...')
                time.sleep(args.sleep_interval)
    # Analyze results
    print_summary(results, args.model_id, args)
    return results


def run_perf_benchmark(args):
    # Check if args is a dictionary or Namespace
    if isinstance(args, dict):
        args = Arguments(**args)
    elif isinstance(args, Namespace):
        args = Arguments.from_args(args)

    if args.seed is not None:
        seed_everything(args.seed)

    # Initialize output directory
    output_path = get_output_path(args)
    configure_logging(args.debug, os.path.join(output_path, 'benchmark.log'))

    # Initialize wandb and swanlab
    if args.wandb_api_key:
        init_wandb(args)
    if args.swanlab_api_key:
        init_swanlab(args)

    # Initialize local server if needed
    if args.api.startswith('local'):
        #  start local server
        server = threading.Thread(target=start_app, args=(copy.deepcopy(args), ), daemon=True)
        server.start()
    # Start benchmark
    if args.auto_parallel or len(args.number) > 1:
        return run_multi_benchmark(args, output_path=output_path)
    else:
        res = []
        res_metric = run_one_benchmark(args, output_path=output_path)
        res.append(res_metric)
        print_summary(res, args.model_id, args)
        return res


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    metrics_result, percentile_result = run_perf_benchmark(args)
    print(metrics_result)
    print(percentile_result)
