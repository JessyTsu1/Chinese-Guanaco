
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArgument():
    model_name_or_path: str = field(
        metadata={
            "required": True,
            "help": "model path"
        }
    )
    tokenizer_name_or_path: str = field(
        metadata={
            "required": True,
            "help": "tokenizer path"
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "required": True,
            "help": (
                "torch dtype"
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )

def main():
    
    model_args, extra_args = HfArgumentParser((
        ModelArgument
    ))
    
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=model_args.torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, use_fast=False)
    
    model.resize_token_embeddings(len(tokenizer))
    
    # 冻结参数
    
if __name__ == "__main__":
    main()