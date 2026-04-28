import os
import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.arguments import parse_args
from utils.logger import setup_logger
from data.loader import load_math_datasets
from data.tokenizer import QwenMathTokenizer

from initialization.base_svd import compute_base_svd
from initialization.fisher_svd import accumulate_fisher_gradients, compute_fisher_svd
from initialization.awq_svd import accumulate_activation_scales, compute_awq_svd
from rl.grpo_extractor import accumulate_rl_fisher_gradients

def main():
    args = parse_args()
    logger = setup_logger("CalibrateSVD")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Model: {args.model_name}")
    # Fix: HuggingFace deprecated torch_dtype in favor of dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tokenizer_handler = QwenMathTokenizer(args.model_name)
    
    from torch.utils.data import default_collate

    raw_dataset = load_math_datasets(args.dataset, split="train[:1000]")
    
    # ---------------------------------------------------------
    # FIX: Conditional Tokenization based on SVD Mode
    # ---------------------------------------------------------
    if args.svd_mode == "rl_fisher":
        # RL needs only prompts (no labels)
        tokenized_dataset = tokenizer_handler.tokenize_rl_dataset(raw_dataset)
        format_cols = ["input_ids", "attention_mask"]
    else:
        # Standard Fisher and AWQ need the full prompt + completion + labels
        tokenized_dataset = tokenizer_handler.tokenize_dataset(raw_dataset)
        format_cols = ["input_ids", "attention_mask", "labels"]

    if "completion" not in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.add_column("completion", raw_dataset["completion"])

    # Set format using the conditionally assigned columns
    tokenized_dataset.set_format(type="torch", columns=format_cols, output_all_columns=True)

    def string_safe_collate(features):
        completions = [f.pop("completion") for f in features]
        batch = default_collate(features)
        batch["completion"] = completions
        return batch

    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=string_safe_collate, pin_memory=True)
    target_layer_names = [name for name, _ in model.named_modules() if "down_proj" in name]

    logger.info(f"Running Calibration Mode: {args.svd_mode}")
    
    if args.svd_mode == "fisher":
        fisher_dict = accumulate_fisher_gradients(model, dataloader, target_layer_names, device)
    elif args.svd_mode == "awq":
        awq_dict = accumulate_activation_scales(model, dataloader, target_layer_names, device)
    elif args.svd_mode == "rl_fisher":
        rl_fisher_dict = accumulate_rl_fisher_gradients(model, tokenizer_handler.tokenizer, dataloader, target_layer_names, num_generations=args.num_generations, device=device)

    os.makedirs(os.path.dirname(args.svd_init_path), exist_ok=True)
    svd_cache = {}
    
    for name in tqdm(target_layer_names, desc="Calibrating SVD per layer"):
        target_weight = model.get_submodule(name).weight.detach()
        
        if args.svd_mode == "baseline":
            U, S, Vh = compute_base_svd(target_weight, rank=args.r_dim)
        elif args.svd_mode == "fisher":
            U, S, Vh = compute_fisher_svd(target_weight, fisher_dict[name], rank=args.r_dim)
        elif args.svd_mode == "awq":
            U, S, Vh = compute_awq_svd(target_weight, awq_dict[name], rank=args.r_dim)
        elif args.svd_mode == "rl_fisher":
            U, S, Vh = compute_fisher_svd(target_weight, rl_fisher_dict[name], rank=args.r_dim)
        else:
            raise ValueError("Invalid svd_mode.")
            
        svd_cache[name] = {"U_r": U, "Sigma_r": S, "V_r": Vh}
        
    torch.save(svd_cache, args.svd_init_path)
    logger.info(f"Saved SVD mapped components to {args.svd_init_path}")

if __name__ == "__main__":
    main()