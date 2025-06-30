import torch
import json
import argparse
import os
import pandas as pd
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
#==========
# Parse Arguments
#==========
parser = argparse.ArgumentParser(description='Args for experiments')
parser.add_argument('--experiment_name',default='all_cod_qwen',type=str,
    help='experiment_name: Selects the experiment which will be evaluated')
parser.add_argument('--rerun',default='yes',type=str,
    help='If it is set to "yes" then search for reurn file instead of output file tensor.')

args = parser.parse_args()
experiment_name = args.experiment_name
rerun = args.rerun

#experiment_path = os.path.join('experiments', experiment_name)
experiment_path = os.path.join('results', 'accuracy_exp', experiment_name)

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
#TODO bug here I think, second rerun tensor not loaded 

if rerun == "yes":
     reruns = []
     for filename in os.listdir(experiment_path):
          if filename.startswith("output") and filename.endswith(".pt"):
               output_tensor_path = os.path.join(experiment_path, filename)
          if filename.startswith("rerun") and filename.endswith(".pt"):
               reruns.append(os.path.join(experiment_path, filename))
     
     # Load original output tensor
     results = torch.load(output_tensor_path)

     # Sort by timestamp extracted from filenames, merge old first, move to newer
     reruns.sort(key=lambda path: datetime.strptime(path.split("/")[-1].replace("rerun_output_", "").replace(".pt", ""), "%Y-%m-%d_%H-%M"))

     # Load and merge rerun results
     for rerun_path in reruns:
          rerun_tensor = torch.load(rerun_path)
          results.update(rerun_tensor)  # overwrite buggy samples with rerun results
     print(f"{output_tensor_path=}")
     print(f"Rerun_paths = {reruns}")
else:
     for filename in os.listdir(experiment_path):
          if filename.startswith("output") and filename.endswith(".pt"):
               output_tensor_path = os.path.join(experiment_path, filename)
     results = torch.load(output_tensor_path)
     print(f"{output_tensor_path=}")

#==========
# Checking for duplicates
#==========
from evaluation_utils import check_for_duplicate_questions
duplicate_entries = check_for_duplicate_questions(exp_tensor=results)
if duplicate_entries:
    print("\nDUPLICATE QUESTIONS DETECTED:")
    for question, key1, key2 in duplicate_entries:
        print(f"Question: {question}\nFound in: {key1} and {key2}\n")
else:
    print("No duplicate questions found.")

#print(results)

#==========
# Evaluation
#==========
from evaluation_utils import calculate_accuracy, compute_entropy, get_latency, get_tokens_per_prompt, compute_logtoku_uncertainty
from evaluation_utils import cos_similarity, plot_logtoku_quadrants, plot_cosine_violin, plot_entropy_violin

accuracy, correctness_dict, answer_dict = calculate_accuracy(exp_tensor=results, prompting_technique=prompting_technique)
entropy = compute_entropy(exp_tensor=results, prompting_technique=prompting_technique, normalize=True)
latency_per_prompt = get_latency(exp_tensor=results)
tokens_per_prompt = get_tokens_per_prompt(exp_tensor=results)
logtoku_results = compute_logtoku_uncertainty(exp_tensor=results,prompting_technique=prompting_technique)
cosines = cos_similarity(exp_tensor=results, model_name = model_name, prompting_technique=prompting_technique)

df_answers = pd.DataFrame([(k, v[0], v[1]) for k, v in answer_dict.items()],columns=["prompt_id", "llm_answer", "ground_truth"])

df_correct = pd.DataFrame(list(correctness_dict.items()), columns=["prompt_id", "correct"])

df_entropy = pd.DataFrame(list(entropy.items()), columns=["prompt_id", "entropy"])

df_latency = pd.DataFrame(list(latency_per_prompt.items()), columns=["prompt_id", "latency"])

df_tokens = pd.DataFrame(list(tokens_per_prompt.items()), columns=["prompt_id", "tokens_used"])

df_logtoku = pd.DataFrame.from_dict(logtoku_results, orient='index').reset_index().rename(columns={'index': 'prompt_id'})

df_cosines = pd.DataFrame(list(cosines.items()), columns=["prompt_id", "cosine"])

# Merge all into a single dataframe on 'prompt_id'
df_merged = df_entropy.merge(df_latency, on="prompt_id") \
                      .merge(df_tokens, on="prompt_id") \
                      .merge(df_correct, on="prompt_id") \
                      .merge(df_answers, on="prompt_id") \
                      .merge(df_logtoku, on="prompt_id") \
                      .merge(df_cosines, on = "prompt_id")

df_merged.to_csv(f"{experiment_path}/evaluation_results.csv", index=False)

#plot logtoku quadrants
plot_path = f"{experiment_path}/logtoku_quadrants.png"
plot_logtoku_quadrants(df_merged, output_path=plot_path)
print(f"Saved LogTokU quadrant plot to: {plot_path}")

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

# ===== Entropy over all samples except buggy ones =====
try:
    entropies_list = list(entropy.values())
    cleaned_list = [x for x in entropies_list if x is not None]
    average_entropy = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_entropy = "Bug occured."

# =====Entropy over all correct answered prompts =====
df_correct = df_merged[df_merged["correct"] == "yes"]
if len(df_correct) > 0:
     average_entropy_correct = df_correct["entropy"].mean()
else:
     average_entropy_correct = "no correct samples"

     

#===== Entropy over all incorrect answered prompts =====
df_incorrect = df_merged[df_merged["correct"] == "no"]
if len(df_incorrect) > 0:
     average_entropy_incorrect = df_incorrect["entropy"].mean()
else:
     average_entropy_incorrect = "no correct samples"

#===== Cosine Similarity average =====
try:
    cosine_list = list(cosines.values())
    cleaned_list = [x for x in cosine_list if x is not None]
    average_cosine = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_entropy = "Bug occured." 

#===== Cosine over all correct answered prompts =====
df_correct = df_merged[df_merged["correct"] == "yes"]
if len(df_correct) > 0:
     average_cosine_correct = df_correct["cosine"].mean()
else:
     average_cosine_correct = "no correct samples"

#===== Cosine over all incorrect answered prompts =====
df_incorrect = df_merged[df_merged["correct"] == "no"]
if len(df_incorrect) > 0:
     average_cosine_incorrect = df_incorrect["cosine"].mean()
else:
     average_cosine_incorrect = "no correct samples"

#plot_entropy_violin(df_correct, df_incorrect)

# ===== Tokens used =====
try:
    tokens_used_list = list(tokens_per_prompt.values())
    cleaned_list = [x for x in tokens_used_list if x is not None]
    average_tokens_used = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_tokens_used = "Bug occured."

# ===== Latency =====
try:
    latency_list = list(latency_per_prompt.values())
    cleaned_list = [x for x in latency_list if x is not None]
    average_latency = sum(cleaned_list) / len(cleaned_list)
except ZeroDivisionError:
     average_latency = "Bug occured."

# ===== Average AU and EU over all prompts =====
au_values = df_logtoku["avg_au"].dropna().tolist()
eu_values = df_logtoku["avg_eu"].dropna().tolist()

average_au = sum(au_values) / len(au_values) if au_values else "no valid values"
average_eu = sum(eu_values) / len(eu_values) if eu_values else "no valid values"

# Also optionally log AU/EU separately for correct/incorrect prompts
au_correct = df_merged[df_merged["correct"] == "yes"]["avg_au"].dropna()
eu_correct = df_merged[df_merged["correct"] == "yes"]["avg_eu"].dropna()

au_incorrect = df_merged[df_merged["correct"] == "no"]["avg_au"].dropna()
eu_incorrect = df_merged[df_merged["correct"] == "no"]["avg_eu"].dropna()

average_au_correct = au_correct.mean() if not au_correct.empty else "no correct samples"
average_eu_correct = eu_correct.mean() if not eu_correct.empty else "no correct samples"
average_au_incorrect = au_incorrect.mean() if not au_incorrect.empty else "no incorrect samples"
average_eu_incorrect = eu_incorrect.mean() if not eu_incorrect.empty else "no incorrect samples"     

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
print(f"{average_cosine=}")
print(f"{average_cosine_correct=}")
print(f"{average_cosine_incorrect=}")
print(f"{average_au=}")
print(f"{average_eu=}")

evaluation_summary = {"samples": len(results),
                       "accuracy": accuracy,
                         "average_entropy": average_entropy,
                          "average_entropy_correct_samples": average_entropy_correct,
                           "average_entropy_incorrect_samples": average_entropy_incorrect,
                           "average_cosine": average_cosine,
                            "average_cosine_correct": average_cosine_correct,
                              "average_cosine_incorrect": average_cosine_incorrect,
                                "average_tokens_used": average_tokens_used,
                                 "average_latency": average_latency,
                                   "average_au": average_au,
                                     "average_eu": average_eu,
                                       "average_au_correct": average_au_correct,
                                        "average_eu_correct": average_eu_correct,
                                         "average_au_incorrect": average_au_incorrect,
                                           "average_eu_incorrect": average_eu_incorrect,}

with open(f"{experiment_path}/evaluation_summary.json", "w") as f:
        json.dump(evaluation_summary, f, indent=4)

print(f"Saved Evaluation Values to: {experiment_path}/evaluation_results.csv")
print(f"Saved Evaluation Summary to: {experiment_path}/evaluation_summary.json")

