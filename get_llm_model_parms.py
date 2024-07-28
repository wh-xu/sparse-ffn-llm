import torch
from torchsummary import summary

import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        print(f"{name}\t{params}")
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
    

# model_name_or_path = "mistralai/Mistral-7B-v0.1"
model_name_or_path = "tiiuae/falcon-7b"
# model_name_or_path = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_config(config)

print(model)
count_parameters(model)
        
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

print(summary(model, input_size=(1,1)))
# from_pretrained(
#     model_name_or_path,
#     cache_dir="./models",
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
# )
