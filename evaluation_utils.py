import torch
import re

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
    if "." in ground_truth or "," in ground_truth:
        return float(ground_truth)
    else:
        return int(ground_truth)

def get_llm_answer(prompt:dict, prompting_technique:str)->float:
    try:
        raw_answer = ''.join(prompt['decoded_tokens'])
    except KeyError:
        return None, None
    
    if prompting_technique == "baseline": #few-shot
        answer = raw_answer
        match = re.search(r"A:\s*(.*?)\s*<eos", answer)
        answer = match.group(1).strip()
    else:
        try:
            _, answer = raw_answer.split('####')
            answer = answer.replace('<eos','')
        except ValueError:
            return None, None

    answer = answer.strip()
    if "." in answer or "," in answer:
        answer = float(answer)
    else:
        answer = int(answer)
    return raw_answer, answer

def calculate_accuracy(exp_tensor:torch.tensor, prompting_technique:str)->float:
    correct_samples = 0
    incorrect_samples = 0
    buggy_sample = 0
    correctness_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        #extract the ground truth answer
        numeric_ground_truth = get_ground_truth(prompt)
        if numeric_ground_truth is None:
            buggy_sample +=1
            continue
        

        #extract the generated answer by the LLM
        _, numeric_answer = get_llm_answer(prompt, prompting_technique)
        if numeric_answer is None:
            buggy_sample +=1
            correctness_dict[prompt_key] = "buggy"
            continue

        #compare LLM answer and ground truth
        if float(numeric_answer) == float(numeric_ground_truth):
            correct_samples += 1
            correctness_dict[prompt_key] = "yes"
        else:
            incorrect_samples += 1
            correctness_dict[prompt_key] = "no"

    prompt_count = len(exp_tensor) 
    n = prompt_count - buggy_sample
    accuracy = f"{correct_samples } / {n}"
    
    return accuracy, correctness_dict

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


def logit_uncertainty():
    #TODO
    ...

def cos_similarity():
    #TODO
    ...
    
