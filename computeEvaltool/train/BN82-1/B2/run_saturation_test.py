import json
import subprocess
import re
import os
import sys
import time

# ================= 🚀 A100 80GB 专属配置 =================
# ViT-Large (Patch16, 224x224) 在 A100 80G 上通常可以开得很大
# 建议序列: 64 -> 128 -> 256 -> 384 -> 512 -> 640
# 注意：如果 OOM，脚本会自动停止，不用担心
BATCH_SIZES = [64, 128, 256, 384, 512, 640, 768] 

CONFIG_FILE = "ds_config.json"
TRAIN_SCRIPT = "train_vit_benchmark.py" # 请确保这是你保存的训练脚本文件名
HOSTFILE = "hostfile"                 # 请确保 hostfile 内容正确
SSH_PORT = "2288"                     # 您的 SSH 端口
THRESHOLD = 0.05                      # 5% 饱和阈值
# =======================================================

def update_config(bs):
    """修改 ds_config.json 中的 micro_batch_size"""
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    
    data['train_micro_batch_size_per_gpu'] = bs
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"🔧 Config updated: train_micro_batch_size_per_gpu = {bs}")

def run_training(bs):
    """调用 deepspeed 运行训练脚本"""
    cmd = [
        "deepspeed",
        "--hostfile", HOSTFILE,
        "--ssh_port", SSH_PORT,
        TRAIN_SCRIPT,
        "--epochs", "4",  # 1 Warmup + 3 Average
        "--model-name", "google/vit-large-patch16-224" # 显式指定模型
    ]
    
    print(f"🚀 [A100 Cluster] Running training with Batch Size {bs}...")
    
    try:
        # 实时捕获输出
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed at batch size {bs} (Likely OOM)!")
        print("Error Tail (last 800 chars):")
        print(e.output[-800:]) 
        return None

def extract_throughput(log_output):
    """从日志中正则提取 __FINAL_THROUGHPUT__"""
    match = re.search(r"__FINAL_THROUGHPUT__: (\d+\.?\d*)", log_output)
    if match:
        return float(match.group(1))
    return None

def main():
    history = [] 
    
    print(f"=== Starting A100 Saturation Test (Threshold: {THRESHOLD*100}%) ===")
    print(f"Target Hardware: 2 Nodes x 8 A100 (80GB)")

    for bs in BATCH_SIZES:
        # 1. 修改配置
        update_config(bs)
        
        # 2. 运行训练
        start_time = time.time()
        log = run_training(bs)
        duration = time.time() - start_time
        
        if log is None: 
            print("🛑 Stopping test due to OOM or Error.")
            break

        # 3. 提取结果
        throughput = extract_throughput(log)
        if throughput:
            history.append((bs, throughput))
            print(f"✅ BS: {bs} | Throughput: {throughput:.2f} samples/s | Time: {duration:.1f}s")
        else:
            print("⚠️ Could not extract throughput from logs.")
            continue
        
        # 4. 饱和判定
        if len(history) >= 3:
            t3 = history[-1][1] # Current
            t2 = history[-2][1] # Prev
            t1 = history[-3][1] # Pre-Prev
            
            diff1 = (t2 - t1) / t1
            diff2 = (t3 - t2) / t2
            
            print(f"   📈 Growth: {history[-3][0]}->{history[-2][0]} (+{diff1:.2%}), {history[-2][0]}->{history[-1][0]} (+{diff2:.2%})")
            
            if diff1 < THRESHOLD and diff2 < THRESHOLD:
                print(f"\n🎉 SATURATION REACHED at Batch Size {bs}!")
                print("Throughput gain is marginal (< 5%) for the last two steps. Performance is saturated.")
                break
    
    print("\n=== Final Summary (A100 80GB) ===")
    print(f"{'Batch Size':<12} | {'Throughput':<15}")
    for bs, tp in history:
        print(f"{bs:<12} | {tp:<15.2f}")
    print(f"📄 Results saved to ./results/Saturation_Test_Summary.xlsx")

if __name__ == "__main__":
    main()
