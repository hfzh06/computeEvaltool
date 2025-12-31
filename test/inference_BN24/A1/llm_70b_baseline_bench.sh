MODEL_NAME="/data/models/deepseek-70b"
API_URL="http://9.0.2.60:30000/v1/chat/completions"
OUTPUT_DIR="./vllm-deepseek70b-results"
TASK_NAME="deepseek-70b-vllm"
PROMPT_LEN=1024
MAX_TOKENS=256
SEED=42

echo "🚀 Running computeEvaltool test for $MODEL_NAME on $API_URL ..."

uv run computeEvaltool llmeval \
  --model "$MODEL_NAME" \
  --api openai \
  --url "$API_URL" \
  --dataset "random" \
  --tokenizer "/data/models/deepseek-70b/" \
  --number 10 \
  --parallel 1 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --prefix-len 0 \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --name "$TASK_NAME" \
  --outputs-dir "$OUTPUT_DIR" \
  --gpu-num 8 \
  --node-num 4 \
  --tp-size 2 \
  --dp-size 1 \
  --inference-engine vllm \
  --gpu-type A800 

echo "✅ Finished: Results saved to $OUTPUT_DIR"
