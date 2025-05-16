import sys
sys.stdout.reconfigure(line_buffering=True)

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
parser.add_argument('--reasoning_qwen', default=False, type=bool,
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


def generate_with_uncertainty(prompt, top_p=0.9, max_new_tokens=20):
    """
    Generates text from a prompt using a custom autoregressive loop.
    At each generation step, forms the nucleus (top-p) token set,
    retrieves their embeddings, computes average pairwise cosine similarity,
    and then (greedily) appends the highest-probability token.

    Args:
        prompt (str): The initial text prompt.
        top_p (float): Cumulative probability threshold for nucleus sampling.
        max_new_tokens (int): Number of tokens to generate.

    Returns:
        dict: Contains:
            - 'generated_text': The complete generated text.
            - 'step_uncertainties': List of avg. cosine similarities per step.
            - 'candidates': List of candidate lists (token+embedding) per step.
    """
    # Encode prompt and move to device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation_candidates = []
    step_uncertainties = []

    for _ in range(max_new_tokens):
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0, -1]                  # (vocab_size,)
        probs = F.softmax(logits, dim=-1)              # (vocab_size,)

        # sort descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)

        # nucleus mask: include all tokens up to cumulative ≥ top_p
        mask = cumulative <= top_p
        mask[0] = True  # always include highest-prob token
        candidate_indices = sorted_indices[mask]

        # gather candidate embeddings & tokens
        step_candidates = []
        candidate_embeddings = []
        for idx in candidate_indices:
            token_str = tokenizer.decode(idx.unsqueeze(0))
            emb = embedding_layer.weight[int(idx)].detach().cpu()
            step_candidates.append({"token": token_str, "embedding": emb})
            candidate_embeddings.append(emb)

        generation_candidates.append(step_candidates)

        # compute uncertainty = avg pairwise cosine similarity
        avg_cos_sim = average_pairwise_cosine_similarity_torch(candidate_embeddings)
        step_uncertainties.append(avg_cos_sim)

        # greedy selection: take the highest-prob token (sorted_indices[0])
        chosen = sorted_indices[0].unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, chosen.to(device)], dim=-1)

    generated_text = tokenizer.decode(input_ids[0])
    return {
        "generated_text": generated_text,
        "step_uncertainties": step_uncertainties,
        "candidates": generation_candidates,
    }
# ============================================================
# 4. Example Usage & Output Uncertainty Measures
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    full_results_data = {}
    correct = 0 #using for 
    for i in range(n_samples):
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
        
        #print(prompt)
        answer = example["answer"]
        torch.cuda.empty_cache()
        with torch.no_grad():
            res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=0.9, max_tokens=tokens_per_response, device=device)

            #entropies = compute_token_entropies(res["top_p_probs"]) 
            entropies = compute_token_entropies(res["full_probs"]) 
            cosines = compute_avg_cosine_similarities(res["top_p_tokens"], embedding_layer.weight) 

        #print_token_info(res, entropies, cosines, tokenizer)

        data_from_one_prompt = {
            "top_p_tokens": [t.detach().cpu() for t in res["top_p_tokens"]],
            "top_p_probs": [t.detach().cpu() for t in res["top_p_probs"]],
            "top_p_logits": [t.detach().cpu() for t in res["top_p_logits"]],
            "generated_tokens": res["generated_tokens"].detach().cpu(),
            "entropies": entropies,
            "cosines": cosines,
            "prompt": prompt #TODO add generated answer, parse answer (####), add expected answer
        }

        #full output string
        answer_string = ""
        for j in data_from_one_prompt["generated_tokens"]:
            answer_string += f" {tokenizer.decode(j)}"
        #print(f"{answer_string=}")
        match = re.search(r'\{.*?\}', answer_string, re.DOTALL)
        no_json = "false"
        if not match:
            #print(f"No JSON object found in output: {answer_string}")
            answer_json_format = '{"answer": 0.0}' # fallback for when llm thinks too long (qwen at question 9 thinks over 800 tokens). TODO just skip that sample instead?
            no_json = "true"
        else:
            model_answer = match.group(0)
            #print(f"{model_answer=}")
            answer_json_format = model_answer.replace(" ", "")
            answer_json_format = answer_json_format.replace('answer":', 'answer": ')

        #print(f"{answer_json_format=}")
        parse_error = "false"
        try:
            data = json.loads(answer_json_format)
            if "answer" not in data:
                data = {"answer": 0.0}
                parse_error = "true"
        except Exception as e:
            #print(f"Error parsing output.")
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
