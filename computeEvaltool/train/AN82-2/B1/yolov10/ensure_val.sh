#!/usr/bin/env bash
set -e
ROOT=/home/wtc/yolov10/coco128
IMG_DIR=${ROOT}/images
LBL_DIR=${ROOT}/labels

# 检查基本结构
if [ ! -d "$IMG_DIR/train2017" ]; then
  echo "找不到 $IMG_DIR/train2017 ，请确认数据已解压到 $ROOT"
  exit 1
fi

# 如果 val2017 已存在，退出
if [ -d "$IMG_DIR/val2017" ] && [ -d "$LBL_DIR/val2017" ]; then
  echo "val2017 已存在，跳过生成"
  exit 0
fi

mkdir -p $IMG_DIR/val2017 $LBL_DIR/val2017

# 取训练集的前 N 张作为验证集（默认 N=20 或可传入参数）
N=${1:-20}
echo "从 train2017 中复制前 $N 张到 val2017"

count=0
for img in $(ls $IMG_DIR/train2017 | sort); do
  if [ $count -ge $N ]; then
    break
  fi
  cp "$IMG_DIR/train2017/$img" "$IMG_DIR/val2017/"
  base="${img%.*}.txt"
  if [ -f "$LBL_DIR/train2017/$base" ]; then
    cp "$LBL_DIR/train2017/$base" "$LBL_DIR/val2017/"
  fi
  count=$((count+1))
done

echo "已生成 val2017 ($count 张)"

