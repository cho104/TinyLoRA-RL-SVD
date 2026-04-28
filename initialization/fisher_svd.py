import torch
import torch.nn as nn
from tqdm import tqdm

def accumulate_fisher_gradients(model: nn.Module, dataloader, target_layer_names: list, device="cuda"):
    model.eval()
    fisher_dict = {name: torch.zeros_like(model.get_submodule(name).weight, device=device) 
                   for name in target_layer_names}
    
    for name, param in model.named_parameters():
        if any(target in name for target in target_layer_names):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for batch in tqdm(dataloader, desc="Calculating Fisher Information"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        model.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        with torch.no_grad():
            for name in target_layer_names:
                weight_param = model.get_submodule(name).weight
                if weight_param.grad is not None:
                    fisher_dict[name] += (weight_param.grad ** 2) / len(dataloader)
                    
    return fisher_dict

def compute_fisher_svd(weight_matrix: torch.Tensor, fisher_grad_squared: torch.Tensor, rank: int = 2, eps: float = 1e-8) -> tuple:
    W = weight_matrix.float()
    F_channel = fisher_grad_squared.float().sum(dim=0)
    D_F_diag = torch.sqrt(F_channel) + eps
    W_tilde = W * D_F_diag.unsqueeze(0)
    
    # FIX: Guard against PyTorch SVD CUDA deadlock
    W_tilde = torch.nan_to_num(W_tilde, nan=0.0, posinf=0.0, neginf=0.0)
    
    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)
    
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    Vh_F = Vh_r / D_F_diag.unsqueeze(0)
    
    dtype = weight_matrix.dtype
    return U_r.to(dtype), S_r.to(dtype), Vh_F.to(dtype)