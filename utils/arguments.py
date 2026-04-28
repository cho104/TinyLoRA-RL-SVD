from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

@dataclass
class ExperimentArguments:
    """Defines the CLI arguments for the ablation study runs."""
    
    # Model & Data
    model_name: str = field(default="Qwen/Qwen2.5-3B-Instruct", metadata={"help": "Base model path"})
    dataset: str = field(default="gsm8k", metadata={"help": "gsm8k or math500"})
    
    # TinyLoRA Configuration
    u_dim: int = field(default=13, metadata={"help": "Dimension of trainable vector v"})
    r_dim: int = field(default=2, metadata={"help": "Frozen SVD rank"})
    n_tie: int = field(default=40, metadata={"help": "Number of layers to share the v vector across"})
    
    # Initialization & Calibration
    svd_mode: str = field(default="baseline", metadata={"help": "baseline, fisher, awq, or rl_fisher"})
    svd_init_path: Optional[str] = field(default=None, metadata={"help": "Path to cached U and V tensors"})
    
    # Topological Constraints
    topological_constraint: str = field(default="none", metadata={"help": "none, mhc, or skew_symmetric"})
    
    # Training & RL
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate for the v vector"})
    epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=4, metadata={"help": "Per-device train batch size"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Accumulate steps to maintain effective batch size"})
    num_generations: int = field(default=4, metadata={"help": "Number of rollouts per prompt for GRPO/RL"})

def parse_args() -> ExperimentArguments:
    parser = HfArgumentParser((ExperimentArguments,))
    args, = parser.parse_args_into_dataclasses()
    return args