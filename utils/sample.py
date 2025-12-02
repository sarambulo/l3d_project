import torch
from math import log, pi as PI, exp

def get_log_probs(mean, logvar, sample):
    return -0.5 * (
        log(2 * PI) +
        logvar +
        (sample - mean)**2 * torch.exp(-logvar)
    )

def get_sample(mean, var):
    std_dev = torch.sqrt(var)
    eps = torch.randn_like(mean)
    eps = torch.clamp(eps, min=-1.0, max=1.0)
    sample = mean + eps * std_dev
    return sample

def get_sample_and_probs(mean, logvar):
    sample = get_sample(mean, torch.exp(logvar))
    log_probs = get_log_probs(mean, logvar, sample)
    return sample, log_probs