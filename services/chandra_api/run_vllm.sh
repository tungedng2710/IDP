# docker run --runtime nvidia --gpus="device=0" -dit \
#   --name chandra_api \
#   -v /root/.cache/huggingface:/root/.cache/huggingface \
#   -e VLLM_ATTENTION_BACKEND=TORCH_SDPA \
#   -p 7871:8000 \
#   --ipc=host \
#   vllm/vllm-openai:nightly\
#   --model datalab-to/chandra \
#   --no-enforce-eager \
#   --max-num-seqs 32 \
#   --dtype bfloat16 \
#   --max-model-len 5000 \
#   --max_num_batched_tokens 65536 \
#   --served-model-name chandra

docker run --runtime nvidia --gpus="device=0" -dit \
  --name chandra_api \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  -p 7871:8000 \
  --ipc=host \
  vllm/vllm-openai:nightly \
  --model datalab-to/chandra \
  --served-model-name chandra \
  --no-enforce-eager \
  --max-num-seqs 32 \
  --max-model-len 5000 \
  --max_num_batched_tokens 65536 \
  --quantization fp8