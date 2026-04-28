import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TinyLoRALinear(nn.Module):
    def __init__(
        self, 
        base_linear: nn.Linear, 
        U_r: torch.Tensor, 
        Sigma_r: torch.Tensor, 
        V_r: torch.Tensor, 
        u_dim: int = 13, 
        r_dim: int = 2,
        constraint: str = "none"
    ):
        super().__init__()
        self.base_layer = base_linear
        self.base_layer.weight.requires_grad = False
        
        # 1. Distribute singular values for numerical stability
        # tinylora_A = sqrt(Sigma) @ V^T (r x in)
        # tinylora_B = U @ sqrt(Sigma) (out x r)
        sqrt_S = torch.sqrt(Sigma_r)
        self.register_buffer("lora_A", (sqrt_S @ V_r).contiguous())
        self.register_buffer("lora_B", (U_r @ sqrt_S).contiguous())
        
        # 2. Frozen random projection matrices P
        self.register_buffer("P", torch.randn(u_dim, r_dim, r_dim) / (r_dim**0.5))
        
        # 3. Trainable vector v
        self.v = nn.Parameter(torch.zeros(u_dim), requires_grad=True)
        self.constraint = constraint

    def _compute_R(self, active_v: torch.Tensor) -> torch.Tensor:
        """Constructs the core r x r update matrix R."""
        # R = sum_i (v_i * P_i)
        R = torch.einsum("i,ijk->jk", active_v, self.P)
        
        # Apply topological constraints
        if self.constraint == "mhc":
            from .mhc_projection import sinkhorn_knopp_projection
            R = sinkhorn_knopp_projection(R)
        elif self.constraint == "skew_symmetric":
            from .mhc_projection import skew_symmetric_projection
            R = skew_symmetric_projection(R)
            
        return R

    def forward(self, x: torch.Tensor, shared_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pass through the frozen base model
        result = self.base_layer(x)
        
        # Capture previous dtype to ensure final output matches
        previous_dtype = x.dtype
        
        # Apply the tiny adapter via sequential projection
        active_v = shared_v if shared_v is not None else self.v
        R = self._compute_R(active_v)
        
        # --- CRITICAL FIX: Cast R and inputs to ensure matching dtypes ---
        # Ensure R matches the precision of the hidden states (bfloat16)
        R = R.to(x.dtype)
        # Ensure A and B buffers are on the right device/dtype (in case of lazy loading)
        A = self.lora_A.to(x.device, x.dtype)
        B = self.lora_B.to(x.device, x.dtype)
        # -----------------------------------------------------------------
        
        # Computation Path: x -> A -> R -> B
        h = F.linear(x, A)              # (batch, seq, r)
        h = F.linear(h, R)              # (batch, seq, r)
        delta = F.linear(h, B)          # (batch, seq, out)
        
        # Add delta to base result and cast back to ensure stability
        result = result + delta
        return result.to(previous_dtype)