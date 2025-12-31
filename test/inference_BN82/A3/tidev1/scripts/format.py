#!/usr/bin/env python3

import json
import sys


def color_text(text: str, color: str) -> str:
    color_codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    return f"\033[{color_codes[color]}m{text}\033[0m"

def parse_baseline(file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    summary = data["summary"]
    print(color_text("Baseline Benchmark Results:", "blue"))
    header = (
        f"{'Model':<12}"
        f"{'Rate (%)':<16}"
        f"{'Elapsed (s)':<14}"
        f"{'RPS (its/s)':<14}"
        f"{'Avg Lat.(ms)':<20}"
    )
    print(color_text(header, "yellow"))
    for model, data in summary.items():
        stats = data["2"]  
        
        req = stats["requests"]
        succ = stats["success"]
        rate = succ / req * 100
        total_elapsed = stats["total_elapsed"]
        rps = stats["rps"]
        avg_latency = f"{stats['avg_user_latency']:.2f}"

        # Format success rate with color and width
        rate_str = f"{rate:.1f}%"
        color = "green" if rate >= 95 else "red"
        colored_rate = color_text(rate_str.rjust(6), color) + " " * 10  # Adjust for total 16 width

        line = (
            f"{model:<12}"
            f"{colored_rate}"
            f"{total_elapsed:<14.2f}"
            f"{rps:<14.2f}"
            f"{avg_latency:<20}"
        )
        print(line)

def parse_saturation(file_path):
    print(color_text("Saturation Benchmark Results:", "blue"))
    with open(file_path) as f:
        data = json.load(f)
    
    data = data["summary"]
    # Remove saturation key for detailed results
    if 'saturation' in data:
        del data['saturation']

    # Header with fixed-width alignment
    header = (
        f"{'Model':<14}"
        f"{'Reqs.':<8}"
        f"{'Succ. Rate (%)':<16}"
        f"{'Elapsed (s)':<14}"
        f"{'RPS (its/s)':<14}"
        f"{'Avg Lat.(ms)':<20}"
    )
    print(color_text(header, "yellow"))

    # Rows
    for model, concs in data.items():
        for conc, metrics in concs.items():
            success = metrics['success']
            requests = metrics['requests']
            rate_val = success / requests * 100 if requests else 0
            rate_percent = f"{rate_val:.2f}%"
            color = "green" if rate_val >= 95 else "red"
            colored_percent = color_text(rate_percent, color)
            rate_plain = f"{rate_val:.2f}%"
            colored_rate = f"{colored_percent}"
            
            # Ensure width control with colors
            width = 16
            plain_len = len(rate_plain)
            if plain_len < width:
                padding = " " * (width - plain_len)
                formatted_rate = colored_rate + padding
            else:
                formatted_rate = colored_rate

            total_elapsed = metrics['total_elapsed']
            rps = metrics['rps']
            avg_latency = f"{metrics['p95_user_latency']:.2f}"
            line = (
                f"{model:<14}"
                f"{metrics['conc']:<8}"
                f"{formatted_rate}"
                f"{total_elapsed:<14.2f}"
                f"{(rps if rps is not None else 'N/A'):<14.2f}"
                f"{avg_latency:<20}"
            )
            print(line)

def parse_saturation_points(file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    data = data["summary"]
    if 'saturation' in data:
        print(color_text("Saturation Points:", "blue"))

        header = (
            f"{'Model':<14}"
            f"{'Sat. Reqs.':<14}"
            f"{'Max RPS (its/s)':<14}"
        )
        print(color_text(header, "yellow"))

        for model, (conc, rps) in data['saturation'].items():
            line = (
                f"{model:<14}"
                f"{conc:<14}"
                f"{rps:<14.2f}"
            )
            print(line)

def main():
    if len(sys.argv) < 3:
        print(color_text("Usage: python format.py <benchmark_type> <file_path>", "red"))
        print(color_text("benchmark_type: baseline | saturation | saturation_points", "cyan"))
        sys.exit(1)
    
    benchmark_type = sys.argv[1]
    file_path = sys.argv[2]
    
    if benchmark_type == "baseline":
        parse_baseline(file_path)
    elif benchmark_type == "saturation":
        parse_saturation(file_path)
    elif benchmark_type == "saturation_points":
        parse_saturation_points(file_path)
    else:
        print(color_text("Unknown benchmark type. Use: baseline | saturation | saturation_points", "red"))
        sys.exit(1)
        
if __name__ == "__main__":
    main()