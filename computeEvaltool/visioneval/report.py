from pathlib import Path
from typing import List, Tuple
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

from .benchmark import calculate_percentiles

logger = logging.getLogger('vision_benchmark')


def print_benchmark_summary(
    all_results: List[Tuple[List[float], List[float], int]],
    model_name: str,
    concurrency: int,
    timeout: int,
    output_dir: Path = None
):
    """打印并保存benchmark摘要报告"""
    console = Console(width=120)
    
    # 创建标题面板
    title = Text('Vision API Benchmark Summary Report', style='bold magenta')
    console.print(Panel(title, width=80))
    
    # 计算总体统计
    total_requests = sum(r[2] for r in all_results)
    all_elapseds = [e for r in all_results for e in r[0]]
    all_responsetime = [rt for r in all_results for rt in r[1]]
    
    avg_elapsed = sum(all_elapseds) / len(all_elapseds) if all_elapseds else 0
    avg_responsetime = sum(all_responsetime) / len(all_responsetime) if all_responsetime else 0
    total_rps = sum(r[2] / timeout for r in all_results)
    
    # 计算百分位数
    elapsed_percentiles = calculate_percentiles(all_elapseds)
    response_percentiles = calculate_percentiles(all_responsetime)
    
    # 基本信息表
    basic_info = Table(show_header=False, width=80, box=None)
    basic_info.add_column('Metric', style='cyan', width=30)
    basic_info.add_column('Value', style='green', width=50)
    
    basic_info.add_row('Model', model_name)
    basic_info.add_row('Test Duration', f'{timeout} seconds')
    basic_info.add_row('Concurrency (per endpoint)', str(concurrency))
    basic_info.add_row('Total Endpoints', str(len(all_results)))
    basic_info.add_row('Total Requests', f'{total_requests:,}')
    basic_info.add_row('Overall RPS', f'{total_rps:.2f}')
    
    console.print('\n📊 Basic Information:')
    console.print(basic_info)
    
    # 详细性能指标表
    table = Table(
        title='📈 Detailed Performance Metrics',
        show_header=True,
        header_style='bold cyan',
        border_style='blue',
        width=120,
    )
    
    table.add_column('Endpoint', justify='left', style='cyan', width=35)
    table.add_column('Requests', justify='right', width=10)
    table.add_column('RPS', justify='right', width=10)
    table.add_column('Avg Lat.(ms)', justify='right', width=15)
    table.add_column('P50 Lat.(ms)', justify='right', width=15)
    table.add_column('P99 Lat.(ms)', justify='right', width=15)
    table.add_column('Avg Resp.(s)', justify='right', width=15)
    
    # 添加每个端点的数据
    for i, (elapseds, responsetimes, requests) in enumerate(all_results):
        endpoint = f"Endpoint {i+1}"
        rps = requests / timeout if timeout > 0 else 0
        avg_lat = sum(elapseds) / len(elapseds) if elapseds else 0
        perc = calculate_percentiles(elapseds)
        avg_resp = sum(responsetimes) / len(responsetimes) if responsetimes else 0
        
        row_style = 'green' if requests > 0 else 'red'
        table.add_row(
            endpoint,
            str(requests),
            f'{rps:.2f}',
            f'{avg_lat:.3f}',
            f'{perc["p50"]:.3f}',
            f'{perc["p99"]:.3f}',
            f'{avg_resp:.3f}',
            style=row_style
        )
    
    console.print('\n')
    console.print(table)
    
    # 总体统计表
    summary_table = Table(
        title='📋 Overall Statistics',
        show_header=True,
        header_style='bold yellow',
        border_style='green',
        width=100,
    )
    
    summary_table.add_column('Metric', justify='left', style='cyan', width=30)
    summary_table.add_column('Latency (ms)', justify='right', width=20)
    summary_table.add_column('Response Time (s)', justify='right', width=20)
    
    summary_table.add_row('Average', f'{avg_elapsed:.3f}', f'{avg_responsetime:.3f}')
    summary_table.add_row('P50', f'{elapsed_percentiles["p50"]:.3f}', f'{response_percentiles["p50"]:.3f}')
    summary_table.add_row('P90', f'{elapsed_percentiles["p90"]:.3f}', f'{response_percentiles["p90"]:.3f}')
    summary_table.add_row('P95', f'{elapsed_percentiles["p95"]:.3f}', f'{response_percentiles["p95"]:.3f}')
    summary_table.add_row('P99', f'{elapsed_percentiles["p99"]:.3f}', f'{response_percentiles["p99"]:.3f}', style='bold')
    
    console.print('\n')
    console.print(summary_table)
    
    # 性能建议
    recommendations = []
    if avg_elapsed > 500:
        recommendations.append('⚠️  Average latency is high (>500ms), consider optimizing model inference')
    if elapsed_percentiles['p99'] > 1000:
        recommendations.append('⚠️  P99 latency exceeds 1s, check for performance bottlenecks')
    if total_rps < len(all_results) * concurrency * 0.5:
        recommendations.append('⚠️  RPS is lower than expected, check server capacity')
    if not recommendations:
        recommendations.append('✅ Performance looks good!')
    
    recommend_text = Text('\n💡 Performance Recommendations:', style='bold cyan')
    console.print(recommend_text)
    for rec in recommendations:
        console.print(f'  {rec}', style='yellow')
    
    # 保存到 Excel
    if output_dir:
        _save_to_excel(all_results, model_name, timeout, output_dir)
    
    console.print('\n')


def _save_to_excel(all_results, model_name, timeout, output_dir: Path):
    """保存结果到Excel文件"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_data = []
        for i, (elapseds, responsetimes, requests) in enumerate(all_results):
            rps = requests / timeout if timeout > 0 else 0
            avg_lat = sum(elapseds) / len(elapseds) if elapseds else float('nan')
            perc = calculate_percentiles(elapseds)
            avg_resp = sum(responsetimes) / len(responsetimes) if responsetimes else float('nan')
            
            summary_data.append([
                f"Endpoint {i+1}",
                requests,
                rps,
                avg_lat,
                perc['p50'],
                perc['p99'],
                avg_resp
            ])
        
        columns = ['Endpoint', 'Requests', 'RPS', 'Avg Lat.(ms)', 'P50 Lat.(ms)', 'P99 Lat.(ms)', 'Avg Resp.(s)']
        df = pd.DataFrame(summary_data, columns=columns)
        
        for c in ['Requests']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        for c in ['RPS', 'Avg Lat.(ms)', 'P50 Lat.(ms)', 'P99 Lat.(ms)', 'Avg Resp.(s)']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        excel_path = output_dir / f"{model_name}_vision_benchmark.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="benchmark")
            ws = writer.sheets["benchmark"]
            
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            
            for col_idx, col_name in enumerate(df.columns, start=1):
                col_letter = get_column_letter(col_idx)
                values = [str(col_name)] + [str(v) for v in df[col_name].tolist()]
                max_len = min(max(len(v) for v in values) + 2, 40)
                ws.column_dimensions[col_letter].width = max(10, max_len)
                
                for r in range(2, ws.max_row + 1):
                    ws.cell(row=r, column=col_idx).alignment = Alignment(horizontal="right")
            
            for c in ['RPS', 'Avg Lat.(ms)', 'P50 Lat.(ms)', 'P99 Lat.(ms)', 'Avg Resp.(s)']:
                if c in df.columns:
                    col_idx = df.columns.get_loc(c) + 1
                    for r in range(2, ws.max_row + 1):
                        ws.cell(row=r, column=col_idx).number_format = '0.0000'
        
        logger.info(f"✅ Benchmark results saved to {excel_path}")
    except Exception as e:
        logger.warning(f"❌ Failed to save Excel: {str(e)}")