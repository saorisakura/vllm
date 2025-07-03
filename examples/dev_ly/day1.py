import json
import logging
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024
)

# 生成文本
prompts = [
    "Once upon a time",
    "In a galaxy far, far away"
]
# INFO 06-25 21:16:05 [llm_engine.py:432] init engine (profile, create kv cache, warmup model) took 0.70 seconds
outputs = llm.generate(prompts, sampling_params)
logging.warning(json.dumps(str(outputs), indent=2))