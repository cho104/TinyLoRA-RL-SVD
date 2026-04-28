import torch
import torch.nn as nn
from tqdm import tqdm

def accumulate_activation_scales(model: nn.Module, dataloader, target_layer_names: list, device="cuda"):
    activation_scales = {name: None for name in target_layer_names}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            mean_abs = x.abs().mean(dim=(0, 1))
            if activation_scales[name] is None:
                activation_scales[name] = mean_abs
            else:
                activation_scales[name] += mean_abs
        return hook

    for name, module in model.named_modules():
        if name in target_layer_names:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating AWQ Scales"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch)
            
    for hook in hooks:
        hook.remove()
        
    for name in activation_scales:
        activation_scales[name] /= len(dataloader)
        
    return activation_scales

def compute_awq_svd(weight_matrix: torch.Tensor, activation_scales: torch.Tensor, rank: int = 2, eps: float = 1e-8) -> tuple:
    W = weight_matrix.float()
    s = activation_scales.float() + eps
    W_tilde = W * s.unsqueeze(0)
    
    # FIX: Guard against PyTorch SVD CUDA deadlock
    W_tilde = torch.nan_to_num(W_tilde, nan=0.0, posinf=0.0, neginf=0.0)
    
    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)
    
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    Vh_AWQ = Vh_r / s.unsqueeze(0)
    
    dtype = weight_matrix.dtype
    return U_r.to(dtype), S_r.to(dtype), Vh_AWQ.to(dtype)