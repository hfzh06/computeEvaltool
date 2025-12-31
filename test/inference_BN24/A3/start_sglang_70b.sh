python -m sglang.launch_server \
  --model-path /data/models/deepseek-70b \
  --port 30000 \
  --host 0.0.0.0 \
  --tp 2 \
  --mem-fraction-static 0.9 \
  --load-format dummy \
  --context-length 2048
