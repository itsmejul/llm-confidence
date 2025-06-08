import sys
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from datasets import load_dataset
import torch.nn.functional as F
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"
from utils import generate_with_top_p, generate_with_top_p_corr, compute_avg_cosine_similarities, compute_token_entropies, load_model
from itertools import combinations
import json
import re

import numpy as np
import os

parser = argparse.ArgumentParser(description="Args for experiments")
parser.add_argument('--dataset_name', default="writingprompts",type=str,
    help='dataset_name: Can be either gsm8k or writingprompts or xsum')
parser.add_argument('--n_samples', default=100, type=int,
    help="n_samples: number of prompts from dataset")
parser.add_argument('--max_tokens', default=100, type=int,
    help='max_tokens: max generated tokens per prompt')
parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B-Instruct', type=str,#meta-llama/Llama-3.1-8B-Instruct # Qwen/Qwen3-8B
    help='model_name: Name or path of the huggingface LLM model to use.')
args = parser.parse_args()
n_samples = args.n_samples
dataset_name = args.dataset_name
max_tokens = args.max_tokens
model_name = args.model_name
model_save_name = model_name.rsplit('/', 1)[-1] # The save folder for the model cant contain any slashes

# Load dataset in format for correlation analysis
# This will return a dataset with one column "prompt" 
# We dont use answer columns because some datasets like writingprompts dont have that
from data_prep import prepare_dataset_for_correlation_analysis
dataset = prepare_dataset_for_correlation_analysis("writingprompts", 100)
print(len(dataset))
print(dataset[0])



print(f"Loading model {model_name} from Huggingface on device {device}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, # .bfloat16, is not supported by v100 gpu # faster than float 32
        #device_map="auto", # device, ,
        # Ensure the model config is set to output hidden states and scores
        output_hidden_states=True,
        # This flag makes the generate() method return additional info (see later)
        return_dict_in_generate=True,
    )
if device == "cuda":
    print("moving model to cuda...")
    model.to("cuda")
print("Successfully loaded model.")
# Ensure model is fully initialized
print("Warming up model...")
_ = model(tokenizer("Hello", return_tensors="pt").to(model.device)["input_ids"])
print("Warmup complete.")

# ========== 
# Get the embedding layer of the model
# ==========
# Embedding layer is the layer of the model that takes tokens as inputs and outputs their vector embeddings
# We will use those embeddings and compare their similarities 
embedding_layer = model.get_input_embeddings()


if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    experiment_data = []

    for i, prompt in enumerate(dataset["prompt"]):
        print(f"Prompt {i}") 
    
        torch.cuda.empty_cache()
        with torch.no_grad():
            #res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=0.9, max_tokens=max_tokens, device=device)
            res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=0.9, max_tokens=max_tokens, device=device)
            print(res["full_probs"])
            entropies = compute_token_entropies(res["top_p_probs"])
            print(entropies)
            cosines = compute_avg_cosine_similarities(res["top_p_tokens"], embedding_layer.weight)
        # prompt, full_entropies, top_p_tokens
        prompt_data = {
            "prompt" : prompt,
            "entropies" : entropies,
            "cosines" : cosines,
            "top_p_tokens" : [t.detach().cpu() for t in res["top_p_tokens"]]
        }

        experiment_data.append(prompt_data)
        #print(experiment_data)
        del res
        del prompt_data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("saving to file")
output_file = f"output_{timestamp}.pt"
save_path = "results/" + dataset_name + "/" + model_save_name + "/" + output_file
txt_save_path = "results/" + dataset_name + "/" + model_save_name + "/" + "hyperparams.txt"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(experiment_data, save_path)
os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)

with open(txt_save_path, "w") as f:
    f.write(model_name)
print("saved")
