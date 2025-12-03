import torch
from math import log, pi as PI, exp

def get_log_probs(mean, var, sample):
    return -0.5 * (log(2 * PI) + torch.log(var) + (sample - mean)**2 / var)

def get_sample(mean, var):
    std_dev = torch.sqrt(var)
    eps = torch.randn_like(mean)
    eps = torch.clamp(eps, min=-0.5, max=0.5)
    sample = mean + eps * std_dev
    return sample

def get_sample_and_probs(mean, var, is_scale: bool = False):
    sample = get_sample(mean, var)
    if is_scale:
        sample = sample.clip(2, None)
    log_probs = get_log_probs(mean, var, sample)
    return sample, log_probs

def get_embedding_log_probs(embedding, sample):
    # Parameter logprobs
    mean_indices = torch.tensor([
        0, 1, 2,
        6, 7, 8, 9,
        14, 15, 16,
    ], device=embedding.device)
    var_indices = torch.tensor([
        3, 4, 5,
        10, 11, 12, 13,
        17, 18, 19,
    ], device=embedding.device)
    mean, var = embedding[:, :, mean_indices], embedding[:, :, var_indices]
    type_logits = embedding[:, :, 20:]
    parameters_logprobs = get_log_probs(mean, var, sample[:, :, :10]) # B, T, 10

    # Type logprobs
    type_logprobs = torch.nn.functional.log_softmax(type_logits, dim=-1)
    sampled_types = sample[:, :, 10:].int() # B, T, 1
    type_logprobs = type_logprobs.gather(dim=-1, index=sampled_types) # B, T, 1

    all_logprobs = torch.concat([parameters_logprobs, type_logprobs], dim=-1) # B, T, 11
    all_logprobs = all_logprobs.sum(dim=-1, keepdim=True) # B, T, 1
    
    assert all_logprobs.shape == (embedding.size(0), embedding.size(1), 1)

    return all_logprobs