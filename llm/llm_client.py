from vllm import LLM, SamplingParams
from openai import OpenAI, AzureOpenAI
import os
import json
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(self, model_path: str = None, model_type: str = "local", quant: Optional[str] = None, 
                 tensor_parallel_size: int = 4, sampling_params = None,
                 base_url: Optional[str] = None, openai_params: Optional[Dict[str, Any]] = None, 
                 config_name: Optional[str] = None, config_file: str = "llm/config.json",
                 use_azure: bool = True, azure_endpoint: Optional[str] = None, api_version: Optional[str] = None):
        """
        Initialize LLMClient for either local vLLM models or OpenAI API models.
        
        Args:
            model_path: Path to local model or OpenAI model name
            model_type: "local" for vLLM models, "openai" for (Azure)OpenAI API
            quant: Quantization method for local models
            tensor_parallel_size: Tensor parallel size for local models
            sampling_params: Sampling parameters for local models
            base_url: Custom base URL for OpenAI-compatible APIs
            openai_params: Additional parameters for OpenAI chat completions
            config_name: Name of predefined config from config.json
            config_file: Path to configuration file
            use_azure: Whether to use Azure OpenAI instead of regular OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
        """
        # Load from config if specified
        if config_name:
            config = self.load_config(config_file, config_name)
            model_path = model_path or config.get("model_path")
            model_type = config.get("model_type", model_type)
            quant = quant or config.get("quant")
            tensor_parallel_size = config.get("tensor_parallel_size", tensor_parallel_size)
            openai_params = openai_params or config.get("openai_params", {})
            use_azure = config.get("use_azure", use_azure)
            azure_endpoint = azure_endpoint or config.get("azure_endpoint")
            api_version = api_version or config.get("api_version")
            
            # Load sampling params from config
            if not sampling_params and "sampling_params" in config:
                sampling_params = SamplingParams(**config["sampling_params"])
        
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        
        if self.model_type == "local":
            if not model_path:
                raise ValueError("model_path is required for local models")
            self.model = self.load_local_llm(model_path, quant, tensor_parallel_size)
            self.sampling_params = sampling_params or self.default_sampling_params()
            self.openai_client = None
            self.openai_params = {}
        elif self.model_type == "openai":
            if not model_path:
                raise ValueError("model_path (OpenAI model name) is required for OpenAI models")
            self.model = None
            self.sampling_params = None
            self.openai_client = self.setup_openai_client(base_url, use_azure, azure_endpoint, api_version)
            self.openai_params = openai_params or self.default_openai_params()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'local' or 'openai'")

    @staticmethod
    def load_config(config_file, config_name) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                configs = json.load(f)
            return configs["model_configs"][config_name]
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Configuration '{config_name}' not found in {config_file}: {e}")

    @classmethod
    def from_config(cls, config_name, config_file="llm/config.json", **kwargs):
        """Create LLMClient instance from predefined configuration"""
        return cls(config_name=config_name, config_file=config_file, **kwargs)

    def load_local_llm(self, model_path: str, quant: Optional[str], tensor_parallel_size: int):
        """Load local VLLM model"""
        return LLM(
            model=model_path,
            enforce_eager=False,  # can set to true for faster inference
            tensor_parallel_size=tensor_parallel_size,
            quantization=quant,
            max_model_len=8192
        )

    def setup_openai_client(self, base_url: Optional[str], 
                          use_azure: bool = True, azure_endpoint: Optional[str] = None, 
                          api_version: Optional[str] = None):
        """Setup OpenAI or Azure OpenAI client"""
        if use_azure:
            return AzureOpenAI(
                azure_endpoint=azure_endpoint or "https://gpt4-endoscribe.openai.azure.com/",
                api_version=api_version or "2025-03-01-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=base_url
            )

    def chat(self, messages: List[Dict[str, str]]) -> Union[List[Any], str]:
        """
        Send chat messages to the model and return response.
        Args:
            messages: List of message dicts with "role" and "content" keys
        Returns:
            For local models: vLLM RequestOutput objects
            For OpenAI models: String response content
        """
        if self.model_type == "local":
            return self.model.chat(messages, self.sampling_params)
        elif self.model_type == "openai":
            return self._chat_openai(messages)

    def _chat_openai(self, messages: List[Dict[str, str]]) -> str:
        """Handle OpenAI API chat completion"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                **self.openai_params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def default_sampling_params(self):
        """Default sampling parameters for local VLLM models"""
        return SamplingParams(
            max_tokens=8000,
            # top_k=1, # greedy. 5/26: can get trapped in invalid completions/early stopping
            temperature=0.15,
            top_p=0.95,
            stop=["<|eot_id|>"]
        )

    def default_openai_params(self) -> Dict[str, Any]:
        """Default parameters for OpenAI API calls"""
        return {
            "max_tokens": 8000,
            "temperature": 0.15,
            "top_p": 0.95,
        }