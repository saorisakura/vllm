"""
Python 的 `accelerate` 包是 Hugging Face 推出的一个工具库，主要作用是**简化分布式训练和推理的实现流程**，帮助开发者在不同硬件环境（单 GPU、多 GPU、CPU、TPU 等）上高效运行 PyTorch 模型，无需手动编写复杂的分布式代码。


### 核心功能与优势
1. **自动适配硬件环境**  
   无需修改代码，即可在单 GPU、多 GPU（数据并行）、CPU、TPU 等环境中运行，自动检测并利用可用资源。

2. **简化分布式训练**  
   封装了 PyTorch 分布式训练的底层逻辑（如 `torch.distributed`、`DDP` 等），开发者无需手动初始化进程组、管理设备分配等，只需通过简单的 API 即可实现分布式训练。

3. **统一代码接口**  
   无论是单卡还是多卡环境，训练和推理的代码逻辑保持一致，降低了跨环境适配的复杂度。

4. **支持主流模型与框架**  
   与 Hugging Face 的 `transformers`、`datasets` 等库无缝集成，特别适合大语言模型（LLM）、计算机视觉等任务的分布式训练和推理。


### 典型使用场景
- **多 GPU 训练**：在多卡环境下自动实现数据并行，加速模型训练。
- **大模型推理**：在有限资源下高效加载大模型（如通过 `device_map` 自动分配模型到不同设备）。
- **跨环境兼容**：同一套代码可在本地单卡、服务器多卡、TPU 等环境中运行。


### 简单示例
使用 `accelerate` 实现分布式训练的核心代码框架如下：
```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 初始化加速器（自动处理分布式环境）
accelerator = Accelerator()

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# 准备数据（示例输入）
inputs = tokenizer("Hello, world!", return_tensors="pt")
labels = inputs.input_ids.clone()

# 准备模型、数据和优化器（加速器自动处理设备分配）
model, inputs, labels = accelerator.prepare(model, inputs, labels)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练循环
model.train()
for _ in range(3):
    outputs = model(** inputs, labels=labels)
    loss = outputs.loss
    accelerator.backward(loss)  # 自动处理分布式梯度同步
    optimizer.step()
    optimizer.zero_grad()
```

运行时，通过 `accelerate launch` 命令启动，自动适配硬件：
```bash
accelerate launch --num_processes=2 your_script.py  # 使用2个GPU
```


### 总结
`accelerate` 的核心价值在于**降低分布式计算的门槛**，让开发者无需深入掌握 PyTorch 分布式细节，就能轻松实现模型在各种硬件环境下的高效运行，尤其适合大模型训练和推理场景。
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import argparse

def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    # 设置主节点地址和端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组，使用NCCL后端（适合GPU通信）
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def load_model(rank, model_name="facebook/opt-125m"):
    """加载模型和分词器"""
    # 只有主进程加载模型，然后通过DDP广播到其他进程
    if rank == 0:
        print(f"Loading model: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度加速
        device_map={"": rank}       # 将模型放置到当前GPU
    )
    
    # 转换为DDP模型
    model = DDP(model, device_ids=[rank])
    
    return model, tokenizer

def generate_text(model, tokenizer, input_text, rank, max_new_tokens=50):
    """生成文本"""
    # 只有主进程处理输入
    if rank == 0:
        # 编码输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(rank)
        
        # 设置生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 生成文本
        with torch.no_grad():
            outputs = model.module.generate(  # 使用.module访问原始模型
                **inputs,
                generation_config=generation_config
            )
        
        # 解码输出
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    return None

def main(rank, world_size, input_text):
    """主函数"""
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model(rank)
        
        # 生成文本
        result = generate_text(model, tokenizer, input_text, rank)
        
        # 只有主进程打印结果
        if rank == 0:
            print(f"输入: {input_text}")
            print(f"输出: {result}")
            
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简易VLLM实现")
    parser.add_argument("--input", type=str, default="Hello, world! ", help="输入文本")
    args = parser.parse_args()
    
    # 设定GPU数量
    world_size = 2  # 2个GPU
    
    # 使用torch.multiprocessing启动多进程
    torch.multiprocessing.spawn(
        main,
        args=(world_size, args.input),
        nprocs=world_size,
        join=True
    )
