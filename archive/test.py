from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "Qwen/Qwen3-8B"  # or "Qwen/Qwen3-8B"
from datasets import concatenate_datasets, load_dataset  
dataset_name = "openai/gsm8k"
raw_dataset = load_dataset(dataset_name, "main")
dataset = concatenate_datasets([raw_dataset[split] for split in raw_dataset.keys()])

print("Loaded Dataset.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, #"auto",        # or torch.float16
    device_map="auto",          # avoids manual .to("cuda")
    output_hidden_states=True, 
    # This flag makes the generate() method return additional info (see later)
    return_dict_in_generate=True,
)

print("Model loaded.")
