#!/bin/bash
source /home/wtc/anaconda3/bin/activate yolo
# ================= 配置区域 =================
# 待测试的全局 Batch Size 列表
# 建议：从 256 开始，每次翻倍，直到显存溢出 (OOM)
BATCH_SIZE_LIST=(256 512 1024 2048 4096)

# 分布式配置
NODE_RANK=1                # <--- 注意：在第二台机器上运行时改为 1
NNODES=2
GPUS_PER_NODE=8
MASTER_ADDR="10.1.73.17"
MASTER_PORT=29612

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp46s0np0
export NCCL_DEBUG=WARN      # 改为 WARN 减少刷屏，除非出错
# ===========================================

echo "========================================================="
echo "   开始 YOLOv10-S 极限性能测试"
echo "   测试列表: ${BATCH_SIZE_LIST[*]}"
echo "   总 GPU 数: $((NNODES * GPUS_PER_NODE))"
echo "========================================================="

# 循环遍历每个 Batch Size
for BS in "${BATCH_SIZE_LIST[@]}"
do
    echo ""
    echo "---------------------------------------------------------"
    echo "[Testing] Global Batch Size = $BS"
    echo "  (每张卡分摊 Batch: $((BS / (NNODES * GPUS_PER_NODE))) )"
    echo "---------------------------------------------------------"

    # 1. 导出环境变量供 Python 脚本读取
    export BATCH_SIZE=$BS

    # 2. 运行 torchrun
    # 使用 '|| true' 确保即使某个 BS 导致 OOM 报错，脚本也能继续跑下一个（或优雅结束）
    torchrun \
      --nnodes=${NNODES} \
      --node_rank=${NODE_RANK} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      /root/computeEvaltool/train/B1/yolov10/test.py || echo "⚠️  警告: 本轮测试非正常退出 (可能是 OOM)"

    # 3. 清理与冷却
    
    sleep 5
    
    # 可选：强制清理显存残留 (仅在非常容易卡死时开启)
    # pkill -f test_ddp.py
done

echo ""
echo "========================================================="
echo "✅ 所有测试结束！"
echo "   文件存入: /root/computeEvaltool/train/B1/results/"
echo "========================================================="