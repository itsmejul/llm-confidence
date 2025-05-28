from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from datasets import load_dataset
import torch.nn.functional as F
import argparse
device_default = "cuda" if torch.cuda.is_available() else "cpu"
from utils import average_pairwise_cosine_similarity_torch, generate_with_top_p, generate_with_top_p_batch, compute_avg_cosine_similarities, compute_token_entropies, print_token_info
from itertools import combinations
import json
import re

import numpy as np
import os
#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--n_samples',default=-1,type=int,
    help='n_samples: Number of articles from the dataset')
parser.add_argument('--model_name', default='Qwen/Qwen3-8B', type=str,
    help='model_name: Name or path of the huggingface LLM model to use.')
parser.add_argument('--dataset', default='openai/gsm8k', type=str,
    help='Name or path of huggingface dataset to use.')
parser.add_argument('--device', default=device_default, type=str,
    help='Device (cuda, cpu, auto).')
parser.add_argument('--tokens_per_response', default=800, type=int,
    help='Generate n tokens in each response and then cut off')
parser.add_argument('--reasoning_qwen', default=True, type=bool,
                    help="True if qwen3-8b should use reasoning, False if not.")
args = parser.parse_args()
n_samples = args.n_samples
model_name = args.model_name
dataset_name = args.dataset
device = args.device
tokens_per_response = args.tokens_per_response

print(args.reasoning_qwen)

#==========
# Load Dataset
#==========
print(f"Loading Dataset {dataset_name} from Huggingface...")
#dataset = load_dataset(dataset_name, "main")
from datasets import concatenate_datasets

raw_dataset = load_dataset(dataset_name, "main")
dataset = concatenate_datasets([raw_dataset[split] for split in raw_dataset.keys()])

print("Loaded Dataset.")
if n_samples == -1:
    n_samples = len(dataset)
else:
    n_samples = min(n_samples, len(dataset)) # Cap dataset size
print(n_samples)
#==========
# Load model
#==========
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
print("moving model to cuda...")
model.to("cuda")
print("Successfully loaded model.")
print("Warming up...")
#_ = model(tokenizer("test", return_tensors="pt").to(model.device))
print("Warmup done.")
# ========== 
# Get the embedding layer of the model
# ==========
# Embedding layer is the layer of the model that takes tokens as inputs and outputs their vector embeddings
# We will use those embeddings and compare their similarities 
embedding_layer = model.get_input_embeddings()



#if __name__ == "__main__":

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
full_results_data = {}
correct = 0 #using for 
for i in range(n_samples):
    print(f"\n Question {i}")
    example = dataset[i]
    question = example["question"]
    print(question)
    #############################
    prompt = f''' You are a math expert. Solve the question which is below delimited by tripple quotes.
        When you’re done, respond **only** with valid JSON of the form  
        {{"answer": <float>}}  
        Question: """{question}"""
        '''
    
    if "qwen3-8b" in str(args.model_name).lower():
        #print("Using qwen3-8b,", end="")
        messages = [{"role": "user", "content": prompt}]
        if args.reasoning_qwen is True:
            #print(f"with reasoning: {args.reasoning_qwen}.")
            text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
        else:
            #print(f"with reasoning: {args.reasoning_qwen}.")
            text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
        prompt = text
    #############################
    
    print(prompt)
    answer = example["answer"]
