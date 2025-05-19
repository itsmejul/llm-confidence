import sys
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from datasets import load_dataset
import torch.nn.functional as F
import argparse
device_default = "cuda" if torch.cuda.is_available() else "cpu"
from utils import generate_with_top_p, compute_avg_cosine_similarities, compute_token_entropies, load_model
from itertools import combinations
import json
import re

import numpy as np
import os
#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--n_samples',default=300,type=int,
    help='n_samples: Number of articles from the dataset')
parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B-Instruct', type=str,#meta-llama/Llama-3.1-8B-Instruct # Qwen/Qwen3-8B
    help='model_name: Name or path of the huggingface LLM model to use.')
parser.add_argument('--dataset', default='openai/gsm8k', type=str,
    help='Name or path of huggingface dataset to use.')
parser.add_argument('--device', default=device_default, type=str,
    help='Device (cuda, cpu, auto).')
parser.add_argument('--tokens_per_response', default=800, type=int,
    help='Generate n tokens in each response and then cut off')
parser.add_argument('--reasoning_qwen', action='store_true',
                    help="Use reasoning mode for qwen3-8b.")
parser.add_argument('--no_reasoning_qwen', dest='reasoning_qwen', action='store_false')
parser.set_defaults(reasoning_qwen=False)
parser.add_argument('--verbose', action='store_true',
                    help="Print debug statements when set to True.")
parser.set_defaults(verbose=False)
parser.add_argument('--local_dir', default='', type=str,
                    help="Use when loading the model locally / debugging locally.")
args = parser.parse_args()
n_samples = args.n_samples
model_name = args.model_name
dataset_name = args.dataset
device = args.device
tokens_per_response = args.tokens_per_response
verbose = args.verbose
local_dir = args.local_dir
reasoning_qwen = args.reasoning_qwen


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
if local_dir != '':
    model, tokenizer = load_model(model_name,local_dir, output_hidden_states=True, return_dict_in_generate=True)
else:
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

# ============================================================
# 4. Example Usage & Output Uncertainty Measures
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    full_results_data = {}
    correct = 0 #using for accuracy
    for i in range(n_samples):
        if verbose:
            print(f"\n Question {i}")
        example = dataset[i]
        question = example["question"]

        #############################
        prompt = f''' You are a math expert. Solve the question which is below delimited by tripple quotes.
            When you’re done, respond **only** with valid JSON of the form  
            {{"answer": <float>}}  
            Question: """{question}"""
            '''
        
        if "qwen3-8b" in str(args.model_name).lower():
            if verbose:
                print("Using qwen3-8b,", end="")
            messages = [{"role": "user", "content": prompt}]
            if reasoning_qwen is True:
                if verbose:
                    print(f"with reasoning: {reasoning_qwen}.")
                text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
            else:
                if verbose:
                    print(f"with reasoning: {reasoning_qwen}.")
                text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                )
            prompt = text
        #############################
        
        if verbose:
            print(f"{prompt=}")
        answer = example["answer"]
        torch.cuda.empty_cache()
        with torch.no_grad():
            #res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=0.9, max_tokens=tokens_per_response, device=device)
            res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=1, max_tokens=tokens_per_response, device=device)

            #entropies = compute_token_entropies(res["top_p_probs"]) 
            #entropies = compute_token_entropies(res["full_probs"]) 
            #cosines = compute_avg_cosine_similarities(res["top_p_tokens"], embedding_layer.weight) 

        #print_token_info(res, entropies, cosines, tokenizer)

        data_from_one_prompt = {
            "top_p_tokens": [t.detach().cpu() for t in res["top_p_tokens"]],
            "top_p_probs": [t.detach().cpu() for t in res["top_p_probs"]],
            "top_p_logits": [t.detach().cpu() for t in res["top_p_logits"]],
            "generated_tokens": res["generated_tokens"].detach().cpu(),
            "decoded_tokens": res["decoded_tokens"],
            #"entropies": entropies,
            #"cosines": cosines,
            "prompt": prompt, #TODO add generated answer, parse answer (####), add expected answer
            "model": model_name
        }

        #full output string
        answer_string = ""
        for token in data_from_one_prompt["decoded_tokens"]:
            answer_string += token
        if verbose:
            print(f"{answer_string=}")
        match = re.search(r'\{.*?\}', answer_string, re.DOTALL)
        no_json = "false"
        if not match:
            if verbose:
                print(f"No JSON object found in output: {answer_string}")
            answer_json_format = '{"answer": 0.0}' # fallback for when llm thinks too long (qwen at question 9 thinks over 800 tokens). TODO just skip that sample instead?
            no_json = "true"
        else:
            model_answer = match.group(0)
            if verbose:
                print(f"{model_answer=}")
            answer_json_format = model_answer.replace(" ", "")
            answer_json_format = answer_json_format.replace('answer":', 'answer": ')

        if verbose:
            print(f"{answer_json_format=}")
        parse_error = "false"
        try:
            data = json.loads(answer_json_format)
            if "answer" not in data:
                data = {"answer": 0.0}
                parse_error = "true"
        except Exception as e:
            if verbose:
                print(f"Error parsing output.")
            #raise e
            parse_error = "true"
            data = {"answer": 0.0} #fallback
        
        ground_truth_split = answer.split('####')
        ground_truth = ground_truth_split[1].strip()

        try:
            ground_truth = float(ground_truth)
            #print(f"{ground_truth=}")
            model_answer = float(data['answer'])
            #print(f"{model_answer=}")
            data_from_one_prompt["ground_truth"] = ground_truth
        except TypeError:
            #print("TypeError, continue.")
            continue
        if no_json == "true":
            data_from_one_prompt["correct"] = "jsonerror"
        elif parse_error == "true":
            data_from_one_prompt["correct"] = "parseerror"
        elif ground_truth == model_answer:
            correct += 1
            data_from_one_prompt["correct"] = "true"
        else: 
            data_from_one_prompt["correct"] = "false"

        full_results_data[f"prompt{i}"] = data_from_one_prompt
        print(full_results_data)
        del res # to free up memory
        del data_from_one_prompt
        gc.collect()
        torch.cuda.empty_cache()
        print(i)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        output_file = f"output_{timestamp}.pt"
        torch.save(full_results_data, output_file)
        print("saved")

    accuracy = float(correct/n_samples)
    print(f"{accuracy=}")
