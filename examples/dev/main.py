# VLLM_USE_PRECOMPILED=1 pip install --editable .
from vllm import LLM, SamplingParams

def vllm_inference_example():
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  # 控制输出的随机性，0表示确定性输出
        top_p=0.9,        # 核采样参数
        max_tokens=100,   # 最大生成 tokens 数
        stop=["\n", "用户:"]  # 停止符
    )
    
    # 初始化LLM，指定模型路径和参数
    # 模型路径可以是本地路径或Hugging Face Hub上的模型名
    llm = LLM(
        # model="meta-llama/Llama-2-7b-chat-hf",  # 模型名称或路径
        model="facebook/opt-125m",
        tensor_parallel_size=2,                 # 张量并行数量，根据GPU数量调整
        gpu_memory_utilization=0.9,             # GPU内存利用率
        max_num_batched_tokens=4096,            # 批处理的最大tokens数
        max_num_seqs=256                        # 批处理的最大序列数
    )
    
    # 输入提示
    prompts = [
        "用户: 什么是人工智能？\n助手:",
        "用户: 介绍一下机器学习的主要分支\n助手:",
        "用户: 如何学习Python编程？\n助手:"
    ]
    
    # 生成文本
    outputs = llm.generate(prompts, sampling_params) # type: ignore
    
    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"输入: {prompt}")
        print(f"输出: {generated_text}\n")

if __name__ == "__main__":
    vllm_inference_example()
