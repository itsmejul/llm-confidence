import torch
import re
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.cluster import KMeans
from huggingface_hub import HfFolder, whoami
import os
import seaborn as sns


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

def get_llm_answer(prompt:dict, prompting_technique:str, prompt_key:str)->float:
    #prompt_key is just for debugging purposes
    try:
        raw_answer = ''.join(prompt['decoded_tokens'])
    except KeyError:
        return None, None
    
    if prompting_technique == "baseline": #few-shot
        answer = raw_answer
        match = re.search(r"A:\s*(.*?)\s*<eos", answer) #re.search(r'\{\s*["\']answer["\']\s*:\s*[-]?\d+(\.\d+)?\s*\}', chosen_tokens)
        if match:
            answer = match.group(1).strip()
        else:
            match = re.search(r"A:\s*(.*?)\s", answer) #search for answers without <eos e.g. A:205
            if match:
                answer = match.group(1).strip()
            else:
                return None, None
    else: #cot or cod
        try:
            _, answer = raw_answer.split('####')
            answer = answer.replace('<eos','')
        except ValueError:
            return None, None

    answer = answer.strip()
    if "." in answer:
        answer = re.sub(r"[^0-9.]+", "", answer) #remove any unit, only keep numbers
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
        _, numeric_answer = get_llm_answer(prompt, prompting_technique, prompt_key)
        if numeric_answer is None:
            buggy_sample +=1
            correctness_dict[prompt_key] = "buggy"
            answer_dict[prompt_key] = ("None", numeric_ground_truth)
            continue

        #compare LLM answer and ground truth
        try:
            float_numeric_answer = float(numeric_answer)
            float_numeric_ground_truth = float(numeric_ground_truth)
        except OverflowError:
            incorrect_samples += 1
            correctness_dict[prompt_key] = "no"
            continue
        if float_numeric_answer == float_numeric_ground_truth:
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

def get_answer_tokens(prompt:dict, prompting_technique:str, prompt_key:str)->list:
    # identify the answer token indices
        answer_token_indices = []
        _, llm_answer = get_llm_answer(prompt, prompting_technique,prompt_key)
        if llm_answer is None:
            return None
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
        return answer_token_indices


def compute_entropy(exp_tensor: torch.tensor, prompting_technique: str, normalize=False) -> dict:
    entropy_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        # identify the answer token indices
        answer_token_indices = get_answer_tokens(prompt, prompting_technique, prompt_key)
        if answer_token_indices is None:
            entropy_dict[prompt_key] = None
            continue
        
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

def compute_logtoku_uncertainty(exp_tensor: dict, prompting_technique: str) -> dict:
    result_dict = {}
    for prompt_key, prompt in exp_tensor.items():
        logtok_au = []
        logtok_eu = []

        # Extract the LLM answer and match characters to token positions
        _, llm_answer = get_llm_answer(prompt, prompting_technique, prompt_key)
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
                logits = prompt['top_p_logits'][idx]  #Tensor of shape (K,)
                if logits.numel() == 0:
                    continue # no candidates: skip
                alpha = F.relu(logits + 1) #checked
                alpha_0 = torch.sum(alpha) #checked
                au = -torch.sum((alpha / alpha_0) * (torch.special.digamma(alpha + 1) - torch.special.digamma(alpha_0 + 1))) #checked
                eu = len(alpha) / torch.sum(alpha + 1) #checked
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

def plot_logtoku_quadrants(df: pd.DataFrame, output_path: str, n_clusters=4) -> None:
    # 1. Clean & cluster
    df_clean = df.dropna(subset=["avg_au", "avg_eu"]).copy()
    X = df_clean[["avg_au", "avg_eu"]].values
    k = min(len(df_clean), n_clusters)
    kmeans = KMeans(n_clusters=k, random_state=0)
    df_clean["cluster"] = kmeans.fit_predict(X)

    # 2. Map correctness to marker shapes
    marker_map = {"yes": "o", "no": "X", "buggy": "s"}
    df_clean["marker"] = df_clean["correct"].map(marker_map).fillna("o")

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for cluster_label in sorted(df_clean["cluster"].unique()):
        sub_cluster = df_clean[df_clean["cluster"] == cluster_label]
        col = cmap(cluster_label % 10)
        for corr_label, marker in marker_map.items():
            sub = sub_cluster[sub_cluster["correct"] == corr_label]
            if sub.empty:
                continue
            ax.scatter(
                sub["avg_au"],
                sub["avg_eu"],
                color=col,          
                marker=marker,           
                s=50,
                edgecolors="k",
                linewidths=0.5,
                alpha=0.8,
            )

    ax.set_xlabel("Aleatoric Uncertainty (AU)")
    ax.set_ylabel("Epistemic Uncertainty (EU)")
    ax.set_title(f"LogTokU Clusters (k={n_clusters}) + Correctness")

    # Build legends
    # Cluster legend
    cluster_handles = [
        Line2D([0], [0], marker="o", color="w",
               label=f"Cluster {i}",
               markerfacecolor=plt.cm.tab10(i),
               markersize=8)
        for i in sorted(df_clean["cluster"].unique())
    ]
    # Correctness legend
    corr_handles = [
        Line2D([0], [0], marker=mk, color="k", label=lbl.capitalize(),
               linestyle="", markersize=8)
        for lbl, mk in marker_map.items()
    ]

    first_legend = ax.legend(handles=cluster_handles,
                             title="Cluster",
                             loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=corr_handles,
              title="Correctness",
              loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def cos_similarity(model_name:str, exp_tensor: torch.tensor, prompting_technique:str):
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM
    from utils import compute_avg_cosine_similarities
    # Log in to hf
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    HfFolder.save_token(hf_token)
    user = whoami()
    print(f"logged in as {user["name"]}")

    #get the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, # .bfloat16, is not supported by v100 gpu, faster than float 32
        output_hidden_states=True, # Ensure the model config is set to output hidden states and scores
        return_dict_in_generate=True, # This flag makes the generate() method return additional info (see later)
    )
    embedding_layer = model.get_input_embeddings()

    cos_dictionary = dict()
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        answer_token_indices = get_answer_tokens(prompt,prompting_technique, prompt_key)
        if answer_token_indices is None:
            continue
        # compute average entropy over answer tokens
        cos_per_token = []
        for idx in answer_token_indices:
            tokens = prompt['top_p_tokens'][idx]
            cosines = compute_avg_cosine_similarities([tokens], embedding_layer.weight)
            cos_per_token.append(cosines[0])
        if len(cos_per_token) == 0:
            avg_cosine = 0
        else:
            avg_cosine = sum(cos_per_token) / len(cos_per_token)
        cos_dictionary[prompt_key] = avg_cosine 
    return cos_dictionary

'''
def plot_entropy_violin(df_correct, df_incorrect):
    df_correct = df_correct.copy()
    df_incorrect = df_incorrect.copy()

    df_correct = df_correct.drop(columns=['label'], errors='ignore')
    df_incorrect = df_incorrect.drop(columns=['label'], errors='ignore')

    df_correct['label'] = 'correct'
    df_incorrect['label'] = 'incorrect'

    df_combined = pd.concat([df_correct[['entropy', 'label']], df_incorrect[['entropy', 'label']]])
    df_combined['label'] = pd.Categorical(df_combined['label'], categories=['correct', 'incorrect'])

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df_combined, x='label', y='entropy', inner='quartile')
    plt.title('Entropy Distribution')
    plt.tight_layout()
    plt.savefig("entropy_violin.png")
'''
'''
def plot_cosine_violin(df_correct, df_incorrect):
    df_correct = df_correct.copy()
    df_incorrect = df_incorrect.copy()
    df_correct['label'] = 'correct'
    df_incorrect['label'] = 'incorrect'

    df_combined = pd.concat([df_correct[['cosine', 'label']], df_incorrect[['cosine', 'label']]])

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df_combined, x='label', y='cosine', inner='quartile')
    plt.title('Cosine Distribution')
    plt.tight_layout()
    plt.savefig("cosine_violin.png")
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_entropy_violin(df_correct, df_incorrect):
    df_correct = df_correct.copy()
    df_incorrect = df_incorrect.copy()

    entropy_correct = df_correct['entropy'].dropna().values
    entropy_incorrect = df_incorrect['entropy'].dropna().values

    fig, ax = plt.subplots(figsize=(6, 5))

    # Full violin for correct
    parts_correct = ax.violinplot(entropy_correct, positions=[0], showmeans=False, showmedians=True,
                                   showextrema=False, widths=0.9)
    for pc in parts_correct['bodies']:
        verts = pc.get_paths()[0].vertices
        center = verts[:, 0].mean()
        # keep left side only
        verts[:, 0] = np.where(verts[:, 0] > center, center, verts[:, 0])
        pc.set_verts([verts])
        pc.set_alpha(0.6)

    # Full violin for incorrect
    parts_incorrect = ax.violinplot(entropy_incorrect, positions=[0], showmeans=False, showmedians=True,
                                     showextrema=False, widths=0.9)
    for pc in parts_incorrect['bodies']:
        verts = pc.get_paths()[0].vertices
        center = verts[:, 0].mean()
        # keep right side only
        verts[:, 0] = np.where(verts[:, 0] < center, center, verts[:, 0])
        pc.set_verts([verts])
        pc.set_alpha(0.6)

    ax.set_xticks([0])
    ax.set_xticklabels(['Entropy\nCorrect vs Incorrect'])
    ax.set_ylabel('Entropy')
    ax.set_title('Half Violin Plot')

    plt.tight_layout()
    plt.savefig("entropy_half_violin.png")

def plot_cosine_violin(df_correct, df_incorrect):
    df_correct = df_correct.copy()
    df_incorrect = df_incorrect.copy()

    cosine_correct = df_correct['cosine'].dropna().values
    cosine_incorrect = df_incorrect['cosine'].dropna().values

    fig, ax = plt.subplots(figsize=(6, 5))

    # Full violin for correct
    parts_correct = ax.violinplot(cosine_correct, positions=[0], showmeans=True, showmedians=False,
                                   showextrema=True, widths=0.9, side="low")
    #for pc in parts_correct['bodies']:
    #    verts = pc.get_paths()[0].vertices
    #    center = verts[:, 0].mean()
        # keep left side only
    #    verts[:, 0] = np.where(verts[:, 0] > center, center, verts[:, 0])
    #    pc.set_verts([verts])
    #    pc.set_alpha(0.6)

    # Full violin for incorrect
    parts_incorrect = ax.violinplot(cosine_incorrect, positions=[0], showmeans=True, showmedians=False,
                                     showextrema=True, widths=0.9, side="high")
    #for pc in parts_incorrect['bodies']:
    #    verts = pc.get_paths()[0].vertices
    #    center = verts[:, 0].mean()
        # keep right side only
    #    verts[:, 0] = np.where(verts[:, 0] < center, center, verts[:, 0])
    #    pc.set_verts([verts])
    #    pc.set_alpha(0.6)

    ax.set_xticks([0])
    ax.set_xticklabels(['Cosine\nCorrect vs Incorrect'])
    ax.set_ylabel('Cosine')
    ax.set_title('Half Violin Plot')

    plt.tight_layout()
    plt.savefig("cosine_half_violin.png")