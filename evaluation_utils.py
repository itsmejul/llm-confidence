import torch

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
    return float(ground_truth)

def get_llm_answer(prompt:dict)->float:
    try:
        raw_answer = ''.join(prompt['decoded_tokens'])
    except KeyError:
        return None
    
    try:
        _, answer = raw_answer.split('####')
    except ValueError:
        return None

    answer = answer.strip()
    return raw_answer, float(answer)

def calculate_accuracy(exp_tensor:torch.tensor)->float:
    correct_samples = 0
    incorrect_samples = 0
    buggy_sample = 0
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]
        #extract the ground truth answer
        numeric_ground_truth = get_ground_truth(prompt)
        if numeric_ground_truth is None:
            buggy_sample +=1
            continue
        

        #extract the generated answer by the LLM
        _, numeric_answer = get_llm_answer(prompt)
        if numeric_answer is None:
            buggy_sample +=1
            continue

        #compare LLM answer and ground truth
        if numeric_answer == numeric_ground_truth:
            correct_samples += 1
        else:
            incorrect_samples += 1

    prompt_count = len(exp_tensor) 
    n = prompt_count - buggy_sample
    accuracy = 0
    try:
        accuracy = correct_samples / n
    except ZeroDivisionError:
        #in this case all samples would be buggy
        return accuracy
    
    return accuracy

def compute_entropy(exp_tensor, normalize=False)->dict:
    entropy_dict = {}
    for prompt_key in exp_tensor.keys():
        prompt = exp_tensor[prompt_key]

        #identify the answer token indices
        answer_token_indices = []
        raw_answer, llm_answer = get_llm_answer(prompt)
        llm_answer = str(llm_answer)
        reversed_raw_answer = raw_answer[::-1]
        for char_llm_answer in llm_answer:
            for idx, char_reversed_raw_answer in enumerate(reversed_raw_answer):
                if char_llm_answer == char_reversed_raw_answer:
                    indice = len(reversed_raw_answer) - idx -1
                    answer_token_indices.append(indice)
                    break
        
        #compute average entropy over answer tokens
        entropy_per_token = []
        for idx in answer_token_indices:
            token_probs = prompt['top_p_probs'][idx]
            #compute entropy
            eps = 1e-12
            probs = token_probs / (token_probs.sum() + eps)
            entropy = -torch.sum(probs * torch.log(probs + eps)).item()
            if normalize:
                entropy /= torch.log(torch.tensor(len(probs)) + eps).item()
            entropy_per_token.append(entropy)
        
        #average entropy of answer tokens
        sum_entropy = sum(entropy_per_token)
        n = len(entropy_per_token)
        average_entropy = sum_entropy / n
        entropy_dict[prompt_key] = average_entropy

    return entropy_dict

def logit_uncertainty():
    ...

def cos_similarity():
    ...
    
