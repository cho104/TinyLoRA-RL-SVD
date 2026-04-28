import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.arguments import parse_args
from utils.logger import setup_logger
from data.loader import load_math_datasets
from data.tokenizer import QwenMathTokenizer
from core.tinylora import TinyLoRALinear

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "completion"}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def replace_with_tinylora(model, target_layers, svd_components, args):
    global_v = nn.Parameter(torch.zeros(args.u_dim)) if args.n_tie > 1 else None
    
    for name, module in model.named_modules():
        if any(target in name for target in target_layers):
            if name not in svd_components:
                continue
                
            layer_svd = svd_components[name]
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            
            tiny_layer = TinyLoRALinear(
                base_linear=module,
                U_r=layer_svd["U_r"], Sigma_r=layer_svd["Sigma_r"], V_r=layer_svd["V_r"],
                u_dim=args.u_dim, r_dim=args.r_dim,
                constraint=args.topological_constraint
            )
            
            if global_v is not None:
                tiny_layer.v = global_v
                
            setattr(parent_module, child_name, tiny_layer)
            
    return model, global_v

def main():
    args = parse_args()
    logger = setup_logger("TrainSFT")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_steps = 100 
    run_id = f"SFT_{args.topological_constraint}_u{args.u_dim}_tie{args.n_tie}_{int(time.time())}"
    log_dir = os.path.join("outputs/runs", run_id)
    checkpoint_dir = os.path.join("outputs/checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Logging SFT metrics to: {log_dir}")

    tokenizer_handler = QwenMathTokenizer(args.model_name)
    train_raw = load_math_datasets(args.dataset, split="train")
    val_raw = load_math_datasets(args.dataset, split="test[:200]") 

    train_tokenized = tokenizer_handler.tokenize_dataset(train_raw)
    val_tokenized = tokenizer_handler.tokenize_dataset(val_raw)
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_tokenized, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_tokenized, batch_size=args.batch_size, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    svd_components = torch.load(args.svd_init_path, map_location=device)
    model, global_v = replace_with_tinylora(model, ["down_proj"], svd_components, args)
    model.to(device)

    trainable_params = [global_v] if args.n_tie > 1 else [p for n, p in model.named_parameters() if "v" in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items() if k != "completion"}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            global_step += 1
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)

            if global_step % eval_steps == 0:
                avg_val_loss = validate(model, val_loader, device)
                writer.add_scalar("Loss/Val_Step", avg_val_loss, global_step)
                logger.info(f"Step {global_step}: Val Loss {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # FIX: Strict Standardization of checkpt keys
                    save_data = {"v": global_v.data if args.n_tie > 1 else {n: p.data for n, p in model.named_parameters() if "v" in n}}
                    torch.save(save_data, os.path.join(checkpoint_dir, "best_model.pt"))
                    logger.info(f"New best SFT model saved at step {global_step}!")
                
                model.train()

    writer.close()
    logger.info("Training SFT complete.")

if __name__ == "__main__":
    main()