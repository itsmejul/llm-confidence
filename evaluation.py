import torch
import json
import argparse
import os
import pandas as pd

#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--experiment_name',default='few_shot_all',type=str,
    help='experiment_name: Selects the experiment which will be evaluated')

args = parser.parse_args()
experiment_name = args.experiment_name

experiment_path = os.path.join('experiments', experiment_name)

#==========
# Metadata
#==========
with open(f"{experiment_path}/metadata.json", "r") as f:
    metadata = json.load(f)

model_name = metadata["model"]
dataset = metadata["dataset"]
prompting_technique = metadata['prompting_technique']

#==========
# Result Tensor
#==========
for filename in os.listdir(experiment_path):
        if filename.startswith("output") and filename.endswith(".pt"):
            output_tensor_path = os.path.join(experiment_path, filename)
results = torch.load(output_tensor_path)

#==========
# Evaluation
#==========
from evaluation_utils import calculate_accuracy, compute_entropy, get_latency, get_tokens_per_prompt

accuracy, correctness_dict, answer_dict = calculate_accuracy(exp_tensor=results, prompting_technique=prompting_technique)
entropy = compute_entropy(exp_tensor=results, prompting_technique=prompting_technique, normalize=True)
latency_per_prompt = get_latency(exp_tensor=results)
tokens_per_prompt = get_tokens_per_prompt(exp_tensor=results)

df_answers = pd.DataFrame([(k, v[0], v[1]) for k, v in answer_dict.items()],columns=["prompt_id", "llm_answer", "ground_truth"])
df_correct = pd.DataFrame(list(correctness_dict.items()), columns=["prompt_id", "correct"])
df_entropy = pd.DataFrame(list(entropy.items()), columns=["prompt_id", "entropy"])
df_latency = pd.DataFrame(list(latency_per_prompt.items()), columns=["prompt_id", "latency"])
df_tokens = pd.DataFrame(list(tokens_per_prompt.items()), columns=["prompt_id", "tokens_used"])

# Merge all into a single dataframe on 'prompt_id'
df_merged = df_entropy.merge(df_latency, on="prompt_id") \
                      .merge(df_tokens, on="prompt_id") \
                      .merge(df_correct, on="prompt_id") \
                      .merge(df_answers, on="prompt_id")
df_merged.to_csv(f"{experiment_path}/evaluation_results.csv", index=False)

#output a list of buggy samples to rerun them later
buggy_samples_indices = []
for key, value in correctness_dict.items():
     if value == "buggy":
          indice = key.replace("prompt", "")
          buggy_samples_indices.append(indice)
df_buggy_indices = pd.DataFrame(buggy_samples_indices, columns=["buggy_prompt_ids"])
df_buggy_indices.to_csv(f"{experiment_path}/buggy_prompts_to_rerun.csv")

#==========
# Compute average values
#==========

#Entropy over all samples except buggy ones
try:
    entropies_list = list(entropy.values())
    cleaned_list = [x for x in entropies_list if x is not None]
    average_entropy = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_entropy = "Bug occured."

#Entropy over all correct answered prompts
df_correct = df_merged[df_merged["correct"] == "yes"]
if len(df_correct) > 0:
     average_entropy_correct = df_correct["entropy"].mean()
else:
     average_entropy_correct = "no correct samples"

     

#Entropy over all incorrect answered prompts
df_incorrect = df_merged[df_merged["correct"] == "no"]
if len(df_incorrect) > 0:
     average_entropy_incorrect = df_incorrect["entropy"].mean()
else:
     average_entropy_incorrect = "no correct samples"

#Tokens used
try:
    tokens_used_list = list(tokens_per_prompt.values())
    cleaned_list = [x for x in tokens_used_list if x is not None]
    average_tokens_used = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_tokens_used = "Bug occured."

#Latency
try:
    latency_list = list(latency_per_prompt.values())
    cleaned_list = [x for x in latency_list if x is not None]
    average_latency = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_latency = "Bug occured."

#==========
# Summary
#==========
print("SUMMARY")
print(f"{model_name=}")
print(f"{dataset=}")
print(f"{prompting_technique}")
print(f"Samples: {len(results)}")
print("---")
print(f"{accuracy=}")
print(f"{average_entropy=}")
print(f"{average_entropy_correct=}")
print(f"{average_entropy_incorrect=}")

evaluation_summary = {"samples": len(results),
                       "accuracy": accuracy,
                         "average_entropy": average_entropy,
                          "average_entropy_correct_samples": average_entropy_correct,
                           "average_entropy_incorrect_samples": average_entropy_incorrect,
                           "average_tokens_used": average_tokens_used,
                             "average_latency": average_latency}

with open(f"{experiment_path}/evaluation_summary.json", "w") as f:
        json.dump(evaluation_summary, f, indent=4)

print(f"Saved Evaluation Values to: {experiment_path}/evaluation_results.csv")
print(f"Saved Evaluation Summary to: {experiment_path}/evaluation_summary.json")

