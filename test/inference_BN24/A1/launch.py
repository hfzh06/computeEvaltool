import os
import subprocess
import signal
import sys
import time
import torch

BASE_PORT = 9000
APP_MODULE = "server:app" # 文件名:FastAPI实例名

def launch_servers():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到任何 GPU，请检查 PyTorch 和 CUDA 环境。")
        sys.exit(1)

    print(f"检测到 {num_gpus} 张 GPU。将为每张卡启动一个服务进程...")
    processes = []
    
    for i in range(num_gpus):
        port = BASE_PORT + i
        # 为每个子进程创建独立的环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        
        command = [
            sys.executable,
            "-m", "uvicorn",
            APP_MODULE,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--workers", "1", # 每个 Uvicorn 实例只开一个 worker
        ]
        
        print(f"正在启动服务: GPU {i} -> http://0.0.0.0:{port}")
        # 使用 Popen 启动非阻塞子进程
        proc = subprocess.Popen(command, env=env)
        processes.append(proc)
        
    print("\n所有服务已启动。按 Ctrl+C 停止所有服务。")
    
    def signal_handler(sig, frame):
        print("\n正在停止所有服务...")
        for p in processes:
            p.terminate() # 发送终止信号
        # 等待所有进程退出
        for p in processes:
            p.wait()
        print("所有服务已停止。")
        sys.exit(0)

    # 捕获 Ctrl+C 信号以实现优雅退出
    signal.signal(signal.SIGINT, signal_handler)
    
    # 让主进程保持运行
    while True:
        time.sleep(1)

if __name__ == "__main__":
    launch_servers()
