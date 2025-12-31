MODEL_NAME="/data/models/deepseek-7b-chat"
API_URL="http://9.0.2.60:18001/v1/chat/completions"
OUTPUT_DIR="./sglang-deepseek7b-results"
TASK_NAME="deepseek-7b-sglang"
PROMPT_LEN=1024
MAX_TOKENS=256
SEED=42

echo "🚀 Running computeEvaltool test for $MODEL_NAME on $API_URL ..."

uv run computeEvaltool llmeval \
  --model "$MODEL_NAME" \
  --api openai \
  --url "$API_URL" \
  --dataset "random" \
  --tokenizer "/data/models/deepseek-7b-chat/" \
  --number 100 \
  --parallel 10 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --prefix-len 0 \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --name "$TASK_NAME" \
  --outputs-dir "$OUTPUT_DIR" \
  --gpu-num 8 \
  --node-num 4 \
  --tp-size 1 \
  --dp-size 2 \
  --inference-engine sglang \
  --gpu-type A800 

echo "✅ Finished: Results saved to $OUTPUT_DIR"
