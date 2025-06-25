import sys
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from datasets import load_dataset
import torch.nn.functional as F
import argparse
device_default = "cpu"
#TODO UNCOMMENT 
#device_default = "cuda" if torch.cuda.is_available() else "cpu" 
from utils import generate_with_top_p, load_model
import json
import yaml
import os
import time
from dotenv import load_dotenv
from huggingface_hub import HfFolder, whoami
import pandas as pd

#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--experiment_name',default='test_llama3',type=str,
    help='experiment_name: Sets the name of the experiment, which will be saved in the experiments/ directory under that name.')
parser.add_argument('--n_samples',default=3,type=int,
    help='n_samples: Number of articles from the dataset')
parser.add_argument('--start_index',default='0',type=int,
    help='start_index: Start index which the dataset questions will be split')
parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B', type=str,#meta-llama/Meta-Llama-3-8B # Qwen/Qwen3-8B, meta-llama/Llama-2-7b-hf
    help='model_name: Name or path of the huggingface LLM model to use.')
parser.add_argument('--dataset', default='openai/gsm8k', type=str,
    help='Name or path of huggingface dataset to use.')
parser.add_argument('--device', default=device_default, type=str,
    help='Device (cuda, cpu, auto).')
parser.add_argument('--tokens_per_response', default=20, type=int,
    help='Generate n tokens in each response and then cut off')
parser.add_argument('--local_dir', default='', type=str,                               
                    help="Use when loading the model locally / debugging locally.")
parser.add_argument('--prompting_technique', default="baseline", type=str,
                    help="Choose a prompting_technique, options are [cot,cod,baseline]. Baseline is the default: plain few-shot examples.")
parser.add_argument('--top_p', default=0.95, type=float,
                    help="Generate tokens whichs probabilities sum up to top_p. If top_p is 1 it will generate ~32k tokenprobs, likely too large.")
parser.add_argument('--rerun_buggy_samples', default="no", type=str,
                    help="If it is set to 'yes': look for an already existing folder with the experiment name and get a list of indices that have to be rerun" \
                    "from a csv buggy_prompts_to_rerun.csv. If it is set to 'no' no change.")

args = parser.parse_args()
experiment_name = args.experiment_name
n_samples = args.n_samples
start_index = args.start_index
model_name = args.model_name
dataset_name = args.dataset
device = args.device
tokens_per_response = args.tokens_per_response
local_dir = args.local_dir
prompting_technique = args.prompting_technique
top_p = args.top_p
rerun_buggy_samples = args.rerun_buggy_samples


#==========
# Log in
#==========
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
HfFolder.save_token(hf_token)
user = whoami()
print(f"logged in as {user["name"]}")

#==========
# Load Dataset
#==========
print(f"Loading Dataset {dataset_name} from Huggingface...")

raw_dataset = load_dataset(dataset_name, "main")
test_dataset = raw_dataset['test'] #Use the test dataset of gsm8k - 1319 samples

if rerun_buggy_samples == "no":
    len_dataset = len(test_dataset['question'])
    if start_index > len_dataset:
        print(f"{start_index=} is bigger than {len_dataset=}")
        raise IndexError

    if n_samples == -1:
        n_samples = len(test_dataset['question'])

    end = start_index + n_samples
    if end > len_dataset:
        end = len(test_dataset['question'])

    questions = test_dataset['question'][start_index:end]
    answers = test_dataset['answer'][start_index:end]
else:
    buggy_csv_path = os.path.join("experiments", experiment_name, "buggy_prompts_to_rerun.csv")
    df_buggy = pd.read_csv(buggy_csv_path)
    buggy_indices = df_buggy["buggy_prompt_ids"].tolist()

    questions = [test_dataset['question'][int(i)] for i in buggy_indices]
    answers = [test_dataset['answer'][int(i)] for i in buggy_indices]
    #note: n_samples is ignored here, it will do all buggy samples

print("Loaded Dataset.")

#==========
# Load model
#==========
print(f"Loading model {model_name} from Huggingface on device {device}...")
if local_dir != '':
    model, tokenizer = load_model(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, # .bfloat16, is not supported by v100 gpu, faster than float 32
        output_hidden_states=True, # Ensure the model config is set to output hidden states and scores
        return_dict_in_generate=True, # This flag makes the generate() method return additional info (see later)
    )

"""local_dir = "/home/max/Studium/Leipzig/Semst…ath_and_ML/hf_models/Qwen/Qwen3-8B/" 
tokenizer = AutoTokenizer.from_pretrained(model_name)                                       
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto", output_hidden_states=True, return_dict_in_generate=True) 
"""

#TODO UNCOMMENT
"""if device == "cuda":
    print("moving model to cuda...")
    model.to("cuda")"""

print("Successfully loaded model.")
# Ensure model is fully initialized
print("Warming up model...")
_ = model(tokenizer("Hello", return_tensors="pt").to(model.device)["input_ids"])
print("Warmup complete.")

# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    full_results_data = {}
    metadata = {"model": model_name, "dataset": dataset_name, "device": device, "experiment_name": experiment_name,
                 "tokens_per_response": tokens_per_response, "prompting_technique":prompting_technique, "top_p": top_p}


    #creat experiment directory
    dir_path = os.path.join("experiments", experiment_name)
    os.makedirs(dir_path, exist_ok=True)

    #save metadata
    if rerun_buggy_samples == "yes":
        metadata_file = os.path.join(dir_path, f"rerun_metadata.json")
    else:
        metadata_file = os.path.join(dir_path, f"metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_file}")
    del metadata #free up memory

    system_prompt = ''
    with open(f"few_shot_examples/gsm8k_{prompting_technique}.yaml", "r") as f:
            prompt_examples = yaml.safe_load(f)
    #system_prompt += "Format:" + prompt_examples['format'] #TODO maybe uncomment again
    for example in prompt_examples['fewshot']:
        system_prompt += "Q: " + example["question"]
        system_prompt += "A:" + example["answer"]
    system_prompt += prompt_examples["system_prompt"]
    
    print("Starting to generate...")
    for i,question in enumerate(questions):
        prompt = system_prompt + "Q: " + question
        print(prompt)
        answer = answers[i]

        torch.cuda.empty_cache()
        with torch.no_grad():
            start_time = time.time()
            res = generate_with_top_p(model=model, tokenizer=tokenizer, prompt=prompt, p=top_p, max_tokens=tokens_per_response, device=device)
            end_time = time.time()
            latency = end_time - start_time

        data_from_one_prompt = {
            "generated_tokens": res["generated_tokens"].detach().cpu(),
            "decoded_tokens": res["decoded_tokens"],
            "top_p_tokens": [t.detach().cpu() for t in res["top_p_tokens"]],
            "top_p_logits": [t.detach().cpu() for t in res["top_p_logits"]],
            "top_p_probs": [t.detach().cpu() for t in res["top_p_probs"]],
            #"entropies": entropies,
            #"cosines": cosines,
            "prompt": prompt,
            "question": question,
            "ground_truth": answer,
            "latency": latency
        }
        if rerun_buggy_samples == "yes":
            full_results_data[f"prompt{buggy_indices[i]}"] = data_from_one_prompt #get the correct indice from the csv list
        else:
            full_results_data[f"prompt{i}"] = data_from_one_prompt

        del res                  # to free up memory
        del data_from_one_prompt # to free up memory
        gc.collect()             # to free up memory
        torch.cuda.empty_cache() # to free up memory
        print(f"Sample {i}: done")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        if rerun_buggy_samples == "yes":
            output_file = os.path.join(dir_path, f"rerun_output_{timestamp}.pt")
        else:
            output_file = os.path.join(dir_path, f"output_{timestamp}.pt")
        torch.save(full_results_data, output_file) #overwrites every sample
        print(f"Saved to {output_file}")
