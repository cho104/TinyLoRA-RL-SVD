import torch

def compute_base_svd(weight_matrix: torch.Tensor, rank: int = 2) -> tuple:
    W_f32 = weight_matrix.float()
    
    # FIX: Guard against PyTorch SVD CUDA deadlock
    W_f32 = torch.nan_to_num(W_f32, nan=0.0, posinf=0.0, neginf=0.0)
    
    U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
    
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    Vh_r = Vh[:rank, :]
    
    dtype = weight_matrix.dtype
    return U_r.to(dtype), S_r.to(dtype), Vh_r.to(dtype)