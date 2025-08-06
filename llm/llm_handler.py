from vllm import LLM
from vllm import SamplingParams


class LLMHandler:
    def __init__(self, model_path, quant=None, tensor_parallel_size=4, sampling_params=None):
        self.model = self.load_llm(model_path, quant, tensor_parallel_size)
        self.sampling_params = sampling_params or self.default_sampling_params()

    def load_llm(self, model_path, quant, tensor_parallel_size):
        return LLM(
            model=model_path,
            enforce_eager=False, # can set to true for faster inference
            tensor_parallel_size=tensor_parallel_size,
            quantization=quant,
            max_model_len=8192
        )

    def chat(self, messages):
        return self.model.chat(messages, self.sampling_params)

    def default_sampling_params(self):
        return SamplingParams(
            max_tokens=8000,
            # top_k=1, # greedy. 5/26: can get trapped in invalid completions/early stopping
            temperature=0.15,
            top_p=0.95,
            stop=["<|eot_id|>"]
        )