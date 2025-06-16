# %% [markdown]
# # Checking the imports

# %%
import torch

print("Is a ROCm-GPU detected? ", torch.cuda.is_available())
print("How many ROCm-GPUs are detected? ", torch.cuda.device_count())

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# %% [markdown]
# # Login to HuggingFace

# %%
# Base model and tokenizer names.
base_model_name = "tiiuae/falcon-7b-instruct"

# Load base model to GPU memory.
device = "cuda:0"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, trust_remote_code=True
).to(device)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
