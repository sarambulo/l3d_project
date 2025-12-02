import torch
from math import log, pi as PI, exp

def get_log_probs(mean, var, sample):
    return -0.5 * (log(2 * PI) + torch.log(var) + (sample - mean)**2 / var)

def get_sample(mean, var):
    std_dev = torch.sqrt(var)
    eps = torch.randn_like(mean)
    eps = torch.clamp(eps, min=-1.0, max=1.0)
    sample = mean + eps * std_dev
    return sample

def get_sample_and_probs(mean, var, is_scale: bool = False):
    sample = get_sample(mean, var)
    if is_scale:
        sample = sample.clip(1e-8, None)
    log_probs = get_log_probs(mean, var, sample)
    return sample, log_probs