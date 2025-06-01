import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


# ==========
# Helper for generating the tokens and saving their logits and probabilities
# ==========
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any


def generate_with_top_p(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    p: float,
    max_tokens: int,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Generate tokens with top-p sampling for a single prompt, tracking each step.

    Returns a dict with:
      - generated_tokens: Tensor of shape (max_tokens,) with generated token IDs
      - top_p_tokens: List[Tensor] of top-p token IDs at each step
      - top_p_logits: List[Tensor] of logits for those tokens
    """
    eos_token_id = tokenizer.eos_token_id
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompt with attention mask
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    generated = []
    top_p_tokens = []
    top_p_logits = []
    top_p_probs = []

    full_logits = []
    full_probs = []

    chosen_tokens = '' #for stopping if <eos> is generated
    for _ in range(max_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :].squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        # save full distribution
        full_logits.append(logits.detach().cpu())
        full_probs.append(probs.detach().cpu())

        # Identify top-p set
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff = torch.searchsorted(cum_probs, p).item() + 1
        top_indices = sorted_indices[:cutoff]
        top_logits = logits[top_indices]
        top_probs = probs[top_indices]

        # Sample
        top_probs_norm = top_probs / top_probs.sum()
        chosen_idx = torch.multinomial(top_probs_norm, 1).item()        
        chosen_token = top_indices[chosen_idx].unsqueeze(0)
        decoded_chosen_tooken = tokenizer.decode(chosen_token)
        chosen_tokens += decoded_chosen_tooken

        # Stop if EOS token is generated
        if eos_token_id is not None and int(chosen_token) == eos_token_id:
            break
        elif "<eos>" in chosen_tokens:
            break

        # Record
        generated.append(chosen_token)
        top_p_tokens.append(top_indices.cpu())
        top_p_logits.append(top_logits.cpu())
        top_p_probs.append(top_probs.cpu())

        # Append for next step
        input_ids = torch.cat([input_ids, chosen_token.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1
        )
    del logits, probs, top_logits, top_probs, sorted_probs, sorted_indices, cum_probs
    torch.cuda.empty_cache()

    answer_tokens = []
    for token in generated:
        answer_tokens.append(tokenizer.decode(token))

    return {
        "generated_tokens": torch.cat(generated, dim=0), #token ids
        "decoded_tokens": answer_tokens, #decoded tokens
        "top_p_tokens": top_p_tokens,
        "top_p_logits": top_p_logits,
        "top_p_probs": top_p_probs,
        #"full_logits" : full_logits,
        #"full_probs" : full_probs,
    }

# ============================================================
# Helper: Average Pairwise Cosine Similarity (PyTorch)
# ============================================================
def average_pairwise_cosine_similarity_torch(vectors):
    """
    Input should be a list of the top-p tokens during generation, for example. 
    It will return the average of all pairwise cosine similarities between those.
    Args:
        vectors (List[torch.Tensor] or torch.Tensor): Either a list of 1D tensors or a 2D tensor of shape (k, d).
    
    Returns:
        float: Average cosine similarity.
    """
    # If provided as a list, stack into a 2D tensor.
    if isinstance(vectors, list):
        vectors = torch.stack(vectors)  # Shape: (k, d)
    
    # Normalize along dimension 1 (for each vector)
    vectors = F.normalize(vectors, p=2, dim=1)
    
    # Compute the full cosine similarity matrix using matrix multiplication.
    similarity_matrix = vectors @ vectors.T  # Shape: (k, k)
    
    # Extract upper triangle (excluding self-similarity) for unique pairs.
    k = vectors.shape[0]
    i, j = torch.triu_indices(k, k, offset=1)
    pairwise_sims = similarity_matrix[i, j]
    
    # Return average similarity as a float.
    return pairwise_sims.mean().item()

def compute_avg_cosine_similarities(token_ids_list, embedding_matrix):
    """
    Compute average pairwise cosine similarities for top-p tokens at each step.

    Args:
        token_ids_list (List[torch.Tensor]): 
            Each tensor contains token IDs (ints) for the top-p candidates at one token position.
        embedding_matrix (torch.Tensor): 
            The model's embedding matrix of shape (vocab_size, hidden_dim).

    Returns:
        List[float]: Average pairwise cosine similarity per token step.
    """
    avg_similarities = []
    for token_ids in token_ids_list:
        embeddings = embedding_matrix[token_ids]  # (top_p, hidden_dim)
        normed = F.normalize(embeddings, p=2, dim=1)
        sims = torch.matmul(normed, normed.T)  # cosine sim matrix
        # Remove diagonal (self-similarities), extract upper triangle
        n = sims.shape[0]
        if n <= 1:
            avg_sim = 1# undefined for 0 or 1 token
        else:
            triu_indices = torch.triu_indices(n, n, offset=1)
            pairwise_sims = sims[triu_indices[0], triu_indices[1]]
            avg_sim = pairwise_sims.mean().item()
        avg_similarities.append(avg_sim)
    return avg_similarities

def compute_token_entropies(probabilities_list):
    entropies = []
    eps = 1e-12
    for probs in probabilities_list:
        probs = probs / (probs.sum() + eps)  # renormalize
        entropy = -torch.sum(probs * torch.log(probs + eps)).item()
        entropies.append(entropy)
    return entropies

def print_token_info(res, entropies, cosines, tokenizer, precision=2):
    token_ids_list = res["top_p_tokens"]
    logits_list = res["top_p_logits"]
    probs_list = res["top_p_probs"]
    entropies_list = entropies
    cosines_list = cosines
    chosen_tokens = tokenizer.convert_ids_to_tokens(res["generated_tokens"])

    assert len(token_ids_list) == len(logits_list) == len(probs_list) == len(chosen_tokens) == len(cosines_list) == len(entropies_list)
    chosen_tokens = tokenizer.convert_ids_to_tokens(res["generated_tokens"])

    num_positions = len(token_ids_list)
    all_triples = []

    # Compute max column widths for alignment
    id_col_width = logit_col_width = prob_col_width = 0
    prefix_col_width = 0
    for i in range(num_positions):
        prefix = f"Token {i+1}: Entr: {entropies_list[i]:.{precision}f}, Cos: {cosines_list[i]:.{precision}f}, {chosen_tokens[i]}"
        prefix_col_width = max(prefix_col_width, len(prefix))
        token_strings = tokenizer.convert_ids_to_tokens(token_ids_list[i])
        triples = []
        for tid, logit, prob in zip(token_strings, logits_list[i], probs_list[i]):
            tid_str = tid.replace("Ġ", " ")

            logit_str = f"{logit:.{precision}f}"
            prob_str = f"{prob * 100:.{precision}f}%"

            triples.append((tid_str, logit_str, prob_str))
            id_col_width = max(id_col_width, len(tid_str))
            logit_col_width = max(logit_col_width, len(logit_str))
            prob_col_width = max(prob_col_width, len(prob_str))
        all_triples.append((prefix, triples))

    # Print aligned
    for prefix, triples in all_triples:
        row = f"{prefix:<{prefix_col_width}}"
        for tid_str, logit_str, prob_str in triples:
            row += f"|{tid_str:>{id_col_width}}, {logit_str:>{logit_col_width}}, {prob_str:>{prob_col_width}}"
        print(row)

def load_model(model_name, local_dir="./models/llama3_70b", output_hidden_states=True, return_dict_in_generate=True):
    import os
    if os.path.exists(local_dir):
        print(f"Loading model from local directory: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype="auto", output_hidden_states=output_hidden_states, return_dict_in_generate=return_dict_in_generate)
    else:
        print(f"Local directory not found. Downloading model '{model_name}' from Hugging Face Hub...")
        """tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", output_hidden_states=True)

        os.makedirs(local_dir, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print(f"Model downloaded and saved locally to: {local_dir}")"""

    return model, tokenizer
