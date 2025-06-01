import torch
import json
import argparse
import os
import pandas as pd

#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--experiment_name',default='cod_test',type=str,
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
from evaluation_utils import calculate_accuracy, compute_entropy

accuracy = calculate_accuracy(exp_tensor=results, prompting_technique=prompting_technique)

entropy = compute_entropy(exp_tensor=results, prompting_technique=prompting_technique, normalize=True)
df = pd.DataFrame(list(entropy.items()), columns=["prompt_id", "entropy"])
df.to_csv(f"{experiment_path}/entropy_results.csv", index=False)

#TODO how many tokens were used, inference time = latency would be interesting too

try:
    entropies_list = list(entropy.values())
    cleaned_list = [x for x in entropies_list if x is not None]
    average_entropy = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_entropy = "Bug occured."


#==========
# Summary
#==========
print("SUMMARY")
print(f"{model_name=}")
print(f"{dataset=}")
print(f"{prompting_technique}")
print("---")
print(f"{accuracy=}")
print(f"Saved Entropies to: {experiment_path}/entropy_results.csv")
print(f"{average_entropy=}")

evaluation_summary = {"accuracy": accuracy, "average_entropy": average_entropy}
with open(f"{experiment_path}/evaluation_summary.json", "w") as f:
        json.dump(evaluation_summary, f, indent=4)
print(f"Saved Evaluation Summary to: {experiment_path}/evaluation_summary.json")

