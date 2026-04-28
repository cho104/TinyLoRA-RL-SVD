import torch
import torch.nn.functional as F

def calculate_activation_kl_divergence(base_acts, tuned_acts):
    base_probs = F.softmax(base_acts.float(), dim=-1)
    tuned_log_probs = F.log_softmax(tuned_acts.float(), dim=-1)
    # batchmean is the mathematically correct KL reduction in PyTorch
    return F.kl_div(tuned_log_probs, base_probs, reduction='batchmean').item()