import torch
import re
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

def get_ground_truth(prompt:dict)->float:
    try:
        raw_ground_truth = prompt['ground_truth']
    except KeyError:
        return None

    try:
        _, ground_truth = raw_ground_truth.split('####')
    except ValueError:
        return None
    
    ground_truth = ground_truth.strip()
    if "." in ground_truth:
        return float(ground_truth)
    else:
        ground_truth = ground_truth.replace(",", "") #convert 2,125 to 2125
        return int(ground_truth)

def get_llm_answer(prompt:dict, prompting_technique:str)->float:
    try:
        raw_answer = ''.join(prompt['decoded_tokens'])
    except KeyError:
        return None, None
    
    if prompting_technique == "baseline": #few-shot
        answer = raw_answer
        match = re.search(r"A:\s*(.*?)\s*<eos", answer)
        if match:
            answer = match.group(1).strip()
        else:
            return None, None
    else:
        try:
            _, answer = raw_answer.split('####')
            answer = answer.replace('<eos','')
        except ValueError:
            return None, None

    answer = answer.strip()
    if "." in answer:
        answer = re.sub(r"[^0-9]+", "", answer) #remove any unit, only keep numbers
        try:
            answer = float(answer)
        except ValueError:
            return None, None
    else:
        answer = answer.replace(",", "") #convert 2,125 to 2125
        answer = re.sub(r"[^0-9]+", "", answer) #remove any unit, only keep numbers
        try:
            answer = int(answer)
        except ValueError:
            return None, None
    return raw_answer, answer

def calculate_accuracy(exp_tensor:torch.tensor, prompting_technique:str)->float:
    correct_samples = 0
    incorrect_samples = 0
    buggy_sample = 0
    correctness_dict = {}
    answer_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        #extract the ground truth answer
        numeric_ground_truth = get_ground_truth(prompt)
        if numeric_ground_truth is None:
            buggy_sample +=1
            correctness_dict[prompt_key] = "buggy"
            answer_dict[prompt_key] = ("None", "None")
            continue
        

        #extract the generated answer by the LLM
        _, numeric_answer = get_llm_answer(prompt, prompting_technique)
        if numeric_answer is None:
            buggy_sample +=1
            correctness_dict[prompt_key] = "buggy"
            answer_dict[prompt_key] = ("None", numeric_ground_truth)
            continue

        #compare LLM answer and ground truth
        if float(numeric_answer) == float(numeric_ground_truth):
            correct_samples += 1
            correctness_dict[prompt_key] = "yes"
        else:
            incorrect_samples += 1
            correctness_dict[prompt_key] = "no"
        
        answer_dict[prompt_key] = (numeric_answer, numeric_ground_truth)

    prompt_count = len(exp_tensor) 
    n = prompt_count - buggy_sample
    accuracy = f"{correct_samples} / {n}" if n > 0 else "0 / 0"
    
    return accuracy, correctness_dict, answer_dict

def compute_entropy(exp_tensor: torch.tensor, prompting_technique: str, normalize=False) -> dict:
    entropy_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]

        # identify the answer token indices
        answer_token_indices = []
        _, llm_answer = get_llm_answer(prompt, prompting_technique)
        if llm_answer is None:
            entropy_dict[prompt_key] = None
            continue
        if isinstance(llm_answer,float): llm_answer = "{:.2f}".format(llm_answer)  # ensure consistent formatting
        else: llm_answer = str(llm_answer)   
        reverse_decoded_tokens = prompt['decoded_tokens'][::-1]
        used_indices = set()

        for char in str(llm_answer[::-1]):  # reverse to match reversed token list
            for idx, token in enumerate(reverse_decoded_tokens):
                real_idx = len(reverse_decoded_tokens) - idx - 1
                if real_idx in used_indices:
                    continue
                if char in token:
                    answer_token_indices.append(real_idx)
                    used_indices.add(real_idx)
                    break
        answer_token_indices = sorted(answer_token_indices)  # to preserve order

        # compute average entropy over answer tokens
        entropy_per_token = []
        for idx in answer_token_indices:
            token_probs = prompt['top_p_probs'][idx]
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-12)).item()
            if normalize:
                denom = torch.log(torch.tensor(len(token_probs))).item()
                if denom > 0: 
                    entropy /= denom
                else:
                    entropy = 0.0 #catches cases where [top_p_probs] has length 1 and then log of 1 is 0.0 -> ZeroDivisionError
            entropy_per_token.append(entropy)

        average_entropy = sum(entropy_per_token) / len(entropy_per_token) if entropy_per_token else 0
        entropy_dict[prompt_key] = average_entropy
    return entropy_dict

def get_latency(exp_tensor: torch.tensor)->dict:
    latency_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        try:
            latency = prompt["latency"]
        except KeyError:
            latency = None
        latency_dict[prompt_key] = latency
    return latency_dict

def get_tokens_per_prompt(exp_tensor: torch.tensor)->dict:
    tokens_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        try:
            tokens = len(prompt["decoded_tokens"])
        except KeyError:
            tokens = None
        tokens_dict[prompt_key] = tokens
    return tokens_dict


def check_for_duplicate_questions(exp_tensor: dict):
    question_to_keys = {}
    duplicates = []

    for key, sample in exp_tensor.items():
        question = sample.get("question", "").strip()
        if question in question_to_keys:
            duplicates.append((question, question_to_keys[question], key))
        else:
            question_to_keys[question] = key

    return duplicates

#TODO check this method, if it properly works and everything is sound
def compute_logtoku_uncertainty(exp_tensor: dict, prompting_technique: str) -> dict:
    result_dict = {}
    for prompt_key, prompt in exp_tensor.items():
        logtok_au = []
        logtok_eu = []

        # Extract the LLM answer and match characters to token positions
        _, llm_answer = get_llm_answer(prompt, prompting_technique)
        if llm_answer is None:
            result_dict[prompt_key] = {"avg_au": None, "avg_eu": None}
            continue

        if isinstance(llm_answer, float):
            llm_answer = "{:.2f}".format(llm_answer)
        else:
            llm_answer = str(llm_answer)

        reverse_decoded_tokens = prompt['decoded_tokens'][::-1]
        used_indices = set()
        answer_token_indices = []

        for char in llm_answer[::-1]:  # reverse to align from end
            for idx, token in enumerate(reverse_decoded_tokens):
                real_idx = len(reverse_decoded_tokens) - idx - 1
                if real_idx in used_indices:
                    continue
                if char in token:
                    answer_token_indices.append(real_idx)
                    used_indices.add(real_idx)
                    break

        answer_token_indices = sorted(answer_token_indices)

        # Compute AU and EU for answer token positions only
        for idx in answer_token_indices:
            try:
                logits = prompt['top_p_logits'][idx]
                if len(logits) < 2:
                    continue
                alpha = F.relu(logits + 1)
                alpha_0 = torch.sum(alpha)
                au = -torch.sum((alpha / alpha_0) * (torch.special.digamma(alpha + 1) - torch.special.digamma(alpha_0 + 1)))
                eu = len(alpha) / torch.sum(alpha + 1)
                logtok_au.append(au.item())
                logtok_eu.append(eu.item())
            except (IndexError, KeyError):
                continue

        if logtok_au and logtok_eu:
            result_dict[prompt_key] = {
                "avg_au": sum(logtok_au) / len(logtok_au),
                "avg_eu": sum(logtok_eu) / len(logtok_eu)
            }
        else:
            result_dict[prompt_key] = {"avg_au": None, "avg_eu": None}
    return result_dict

#TODO check this for soundness
def plot_logtoku_quadrants(df: pd.DataFrame, output_path: str) -> None:
    # ---- 1. filter & normalise ------------------------------------------------
    df_plot = df.dropna(subset=["avg_au", "avg_eu"]).copy()

    def _normalise_series(series: pd.Series) -> pd.Series:
        vmin, vmax = series.min(), series.max()
        if vmax == vmin:
            return series * 0.0
        return (series - vmin) / (vmax - vmin)

    df_plot["norm_au"] = _normalise_series(df_plot["avg_au"])
    df_plot["norm_eu"] = _normalise_series(df_plot["avg_eu"])

    # ---- 2. quadrant classification ------------------------------------------
    def classify_quadrant(row):
        if row["norm_au"] >= 0.5 and row["norm_eu"] >= 0.5:
            return "I"
        elif row["norm_au"] < 0.5 and row["norm_eu"] >= 0.5:
            return "II"
        elif row["norm_au"] < 0.5 and row["norm_eu"] < 0.5:
            return "III"
        else:
            return "IV"

    df_plot["quadrant"] = df_plot.apply(classify_quadrant, axis=1)

    # ---- 3. colour-map for correctness ---------------------------------------
    color_map = {"yes": "green", "no": "red", "buggy": "gray"}
    colors = df_plot["correct"].map(color_map).fillna("gray")

    # ---- 4. plotting ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # quadrant grid
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)

    # scatter
    ax.scatter(
        df_plot["norm_au"],
        df_plot["norm_eu"],
        c=colors,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
        s=40,
    )

    # labels / legend
    ax.set_xlabel("Normalised Aleatoric Uncertainty (AU)")
    ax.set_ylabel("Normalised Epistemic Uncertainty (EU)")
    ax.set_title("LogTokU Quadrants per Prompt")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Correct",
               markerfacecolor="green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Incorrect",
               markerfacecolor="red", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Buggy",
               markerfacecolor="gray", markersize=8),
    ]
    ax.legend(handles=legend_elements, title="Prompt Outcome", loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def cos_similarity():
    #TODO
    ...
    
