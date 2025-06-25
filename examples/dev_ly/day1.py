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
outputs = llm.generate(prompts, sampling_params)
print(outputs)