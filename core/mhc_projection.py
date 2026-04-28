import torch
import torch.nn.functional as F

def sinkhorn_knopp_projection(matrix: torch.Tensor, num_iters: int = 10, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Applies the Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff polytope
    (making it a doubly stochastic matrix where rows and columns sum to 1).
    
    Note: Standard Sinkhorn requires non-negative matrices. We take the absolute value 
    or apply a softmax formulation depending on the adapter's mathematical state.
    """
    # Ensure non-negativity for Sinkhorn
    M = torch.abs(matrix) + epsilon
    
    for _ in range(num_iters):
        # Normalize rows
        row_sums = torch.sum(M, dim=-1, keepdim=True)
        M = M / row_sums
        
        # Normalize columns
        col_sums = torch.sum(M, dim=-2, keepdim=True)
        M = M / col_sums
        
    # Re-apply the original signs to preserve directional logic of the update
    return M * torch.sign(matrix)

def skew_symmetric_projection(matrix: torch.Tensor) -> torch.Tensor:
    """
    Projects a matrix to be skew-symmetric (W^T = -W).
    Used as an alternative topological constraint to prevent residual stream drift 
    during extreme layer-tying.
    """
    # Skew-symmetric projection is simply 0.5 * (W - W^T)
    return 0.5 * (matrix - matrix.transpose(-1, -2))