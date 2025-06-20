from vllm import LLM

llm = LLM(model="facebook/opt-125m")
llm.start_profile()
