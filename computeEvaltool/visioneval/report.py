from pathlib import Path
from typing import Dict, List, Tuple
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .benchmark import calculate_percentiles

logger = logging.getLogger('vision_benchmark')


def print_benchmark_summary(
    all_results: Dict[str, Dict[int, Tuple[List[float], List[float], int]]],
    timeout: int,
    output_dir: Path = None
):
    """打印并保存 benchmark 摘要报告
    
    Args:
        all_results: {model_name: {concurrency: (elapseds, responsetimes, total_requests)}}
    """
    console = Console(width=140)
    
    # 创建标题面板
    title = Text('Vision API Benchmark Summary Report', style='bold magenta')
    console.print(Panel(title, width=100))
    
    # 汇总统计
    total_tests = sum(len(conc_results) for conc_results in all_results.values())
    total_requests_all = sum(
        result[2] 
        for model_results in all_results.values() 
        for result in model_results.values()
    )
    
    # 基本信息表
    basic_info = Table(show_header=False, width=100, box=None)
    basic_info.add_column('Metric', style='cyan', width=35)
    basic_info.add_column('Value', style='green', width=65)
    
    basic_info.add_row('Models Tested', ', '.join(all_results.keys()))
    basic_info.add_row('Test Duration (per test)', f'{timeout} seconds')
    basic_info.add_row('Total Test Configurations', str(total_tests))
    basic_info.add_row('Total Requests (all tests)', f'{total_requests_all:,}')
    
    console.print('\n📊 Basic Information:')
    console.print(basic_info)
    
    # 详细性能指标表
    table = Table(
        title='📈 Detailed Performance Metrics by Model and Concurrency',
        show_header=True,
        header_style='bold cyan',
        border_style='blue',
        width=140,
    )
    
    table.add_column('Model', justify='left', style='cyan', width=15)
    table.add_column('Conc.', justify='right', width=8)
    table.add_column('Requests', justify='right', width=10)
    table.add_column('RPS', justify='right', width=10)
    table.add_column('Avg Lat.(ms)', justify='right', width=13)
    table.add_column('P50 (ms)', justify='right', width=13)
    table.add_column('P90 (ms)', justify='right', width=13)
    table.add_column('P99 (ms)', justify='right', width=13)
    table.add_column('Avg Resp.(s)', justify='right', width=13)
    
    # 按模型分组添加数据
    for model_name, conc_results in sorted(all_results.items()):
        for conc, (elapseds, responsetimes, requests) in sorted(conc_results.items()):
            rps = requests / timeout if timeout > 0 else 0
            avg_lat = sum(elapseds) / len(elapseds) if elapseds else 0
            perc = calculate_percentiles(elapseds)
            avg_resp = sum(responsetimes) / len(responsetimes) if responsetimes else 0
            
            row_style = 'green' if requests > 0 else 'red'
            table.add_row(
                model_name,
                str(conc),
                f'{requests:,}',
                f'{rps:.2f}',
                f'{avg_lat:.3f}',
                f'{perc["p50"]:.3f}',
                f'{perc["p90"]:.3f}',
                f'{perc["p99"]:.3f}',
                f'{avg_resp:.3f}',
                style=row_style
            )
    
    console.print('\n')
    console.print(table)
    
    # 按模型汇总的统计表
    summary_table = Table(
        title='📋 Summary by Model',
        show_header=True,
        header_style='bold yellow',
        border_style='green',
        width=120,
    )
    
    summary_table.add_column('Model', justify='left', style='cyan', width=20)
    summary_table.add_column('Avg RPS', justify='right', width=15)
    summary_table.add_column('Avg Latency (ms)', justify='right', width=20)
    summary_table.add_column('P99 Latency (ms)', justify='right', width=20)
    summary_table.add_column('Total Requests', justify='right', width=20)
    
    for model_name, conc_results in sorted(all_results.items()):
        all_elapseds = [e for result in conc_results.values() for e in result[0]]
        total_requests = sum(result[2] for result in conc_results.values())
        total_duration = timeout * len(conc_results)
        
        avg_rps = total_requests / total_duration if total_duration > 0 else 0
        avg_latency = sum(all_elapseds) / len(all_elapseds) if all_elapseds else 0
        p99 = calculate_percentiles(all_elapseds)['p99']
        
        summary_table.add_row(
            model_name,
            f'{avg_rps:.2f}',
            f'{avg_latency:.3f}',
            f'{p99:.3f}',
            f'{total_requests:,}'
        )
    
    console.print('\n')
    # console.print(summary_table)
    
    # 性能建议
    recommendations = []
    for model_name, conc_results in all_results.items():
        all_elapseds = [e for result in conc_results.values() for e in result[0]]
        avg_lat = sum(all_elapseds) / len(all_elapseds) if all_elapseds else 0
        p99 = calculate_percentiles(all_elapseds)['p99']
        
        if avg_lat > 500:
            recommendations.append(f'⚠️  {model_name}: Average latency is high ({avg_lat:.1f}ms)')
        if p99 > 1000:
            recommendations.append(f'⚠️  {model_name}: P99 latency exceeds 1s ({p99:.1f}ms)')
    
    if not recommendations:
        recommendations.append('✅ All models performance looks good!')
    
    recommend_text = Text('\n💡 Performance Recommendations:', style='bold cyan')
    console.print(recommend_text)
    for rec in recommendations:
        console.print(f'  {rec}', style='yellow')
    
    # 保存到 Excel
    if output_dir:
        _save_to_excel(all_results, timeout, output_dir)
    
    console.print('\n')


def _save_to_excel(all_results: Dict[str, Dict[int, Tuple]], timeout: int, output_dir: Path):
    """保存结果到 Excel 文件"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        summary_data = []
        for model_name, conc_results in sorted(all_results.items()):
            for conc, (elapseds, responsetimes, requests) in sorted(conc_results.items()):
                rps = requests / timeout if timeout > 0 else 0
                avg_lat = sum(elapseds) / len(elapseds) if elapseds else float('nan')
                perc = calculate_percentiles(elapseds)
                avg_resp = sum(responsetimes) / len(responsetimes) if responsetimes else float('nan')
                
                summary_data.append([
                    model_name,
                    conc,
                    requests,
                    rps,
                    avg_lat,
                    perc['p50'],
                    perc['p90'],
                    perc['p99'],
                    avg_resp
                ])
        
        columns = ['Model', 'Concurrency', 'Requests', 'RPS', 'Avg Lat.(ms)', 
                   'P50 (ms)', 'P90 (ms)', 'P99 (ms)', 'Avg Resp.(s)']
        df = pd.DataFrame(summary_data, columns=columns)
        
        # 数据类型转换
        for c in ['Concurrency', 'Requests']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        for c in ['RPS', 'Avg Lat.(ms)', 'P50 (ms)', 'P90 (ms)', 'P99 (ms)', 'Avg Resp.(s)']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        excel_path = output_dir / f"vision_benchmark_multi_model.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="benchmark")
            ws = writer.sheets["benchmark"]
            
            # 冻结首行
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            
            # 表头样式
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for col_idx, col_name in enumerate(df.columns, start=1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                
                # 设置列宽
                col_letter = get_column_letter(col_idx)
                values = [str(col_name)] + [str(v) for v in df[col_name].tolist()]
                max_len = min(max(len(v) for v in values) + 2, 40)
                ws.column_dimensions[col_letter].width = max(12, max_len)
                
                # 数据对齐
                for r in range(2, ws.max_row + 1):
                    ws.cell(row=r, column=col_idx).alignment = Alignment(horizontal="right")
            
            # 数字格式
            for c in ['RPS', 'Avg Lat.(ms)', 'P50 (ms)', 'P90 (ms)', 'P99 (ms)', 'Avg Resp.(s)']:
                if c in df.columns:
                    col_idx = df.columns.get_loc(c) + 1
                    for r in range(2, ws.max_row + 1):
                        ws.cell(row=r, column=col_idx).number_format = '0.000'
        
        logger.info(f"✅ Benchmark results saved to {excel_path}")
    except Exception as e:
        logger.exception(f"❌ Failed to save Excel: {str(e)}")