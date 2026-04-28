import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from utils.arguments import parse_args
from utils.logger import setup_logger
from data.loader import load_math_datasets
from data.tokenizer import QwenMathTokenizer
from scripts.train import replace_with_tinylora
from rl.reward_fns import gsm8k_exact_match_reward

def string_safe_collate(features):
    completions = [f.pop("completion") for f in features]
    batch = default_collate(features)
    batch["completion"] = completions
    return batch

def main():
    args = parse_args()
    logger = setup_logger("TrainRL_GRPO")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = f"RL_{args.topological_constraint}_u{args.u_dim}_tie{args.n_tie}_{int(time.time())}"
    log_dir = os.path.join("outputs/runs", run_id)
    checkpoint_dir = os.path.join("outputs/checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Logging RL metrics to: {log_dir}")

    tokenizer_handler = QwenMathTokenizer(args.model_name)
    raw_dataset = load_math_datasets(args.dataset, split="train")
    tokenized_dataset = tokenizer_handler.tokenize_rl_dataset(raw_dataset)
    
    if "completion" not in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.add_column("completion", raw_dataset["completion"])
        
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    train_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=string_safe_collate)

    # MEMORY OPTIMIZATION 1: Use Flash Attention 2 (Massive VRAM savings)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(device)
    
    # MEMORY OPTIMIZATION 2: Explicitly freeze the base model!
    for param in model.parameters():
        param.requires_grad = False
        
    model.gradient_checkpointing_enable()
    
    svd_components = torch.load(args.svd_init_path, map_location=device)
    model, global_v = replace_with_tinylora(model, ["down_proj"], svd_components, args)
    
    # Ensure ONLY our v parameters require gradients after injection
    trainable_params = []
    for n, p in model.named_parameters():
        if "v" in n.split(".")[-1]: # Target TinyLoRA's 'v'
            p.requires_grad = True
            trainable_params.append(p)
            
    if args.n_tie > 1:
        global_v.requires_grad = True
        trainable_params = [global_v]

    model.to(device)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    optimizer.zero_grad() # Initialize safely

    global_step = 0
    best_reward = 0.0

    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_loader, desc=f"RL Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ground_truths = batch["completion"]
            
            actual_max_len = attention_mask.sum(dim=1).max().item()
            if actual_max_len > 600:
                continue
                
            pad_len = input_ids.shape[1] - actual_max_len
            sliced_input_ids = input_ids[:, pad_len:]
            sliced_attention_mask = attention_mask[:, pad_len:]
                
            model.eval() 
            with torch.no_grad():
                gen_outputs = model.generate(
                    input_ids=sliced_input_ids,
                    attention_mask=sliced_attention_mask,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=args.num_generations,
                    pad_token_id=tokenizer_handler.tokenizer.pad_token_id
                )
                
            gen_texts = tokenizer_handler.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            expanded_truths = [t for t in ground_truths for _ in range(args.num_generations)]
            
            rewards = gsm8k_exact_match_reward(gen_texts, expanded_truths)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            advantages = torch.zeros_like(rewards_tensor)
            for j in range(0, len(rewards_tensor), args.num_generations):
                group_rewards = rewards_tensor[j:j+args.num_generations]
                adv = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
                advantages[j:j+args.num_generations] = adv

            # Free up generation cache before doing the backward pass
            torch.cuda.empty_cache()

            model.train()
            gen_attention_mask = (gen_outputs != tokenizer_handler.tokenizer.pad_token_id).long()
            
            # MEMORY OPTIMIZATION 3: use_cache=False prevents saving KV states during train mode
            outputs = model(
                input_ids=gen_outputs, 
                attention_mask=gen_attention_mask,
                use_cache=False 
            )
            logits = outputs.logits
            del outputs
            
            prompt_length = sliced_input_ids.shape[1]
            gen_logits = logits[:, prompt_length - 1 : -1, :].contiguous()
            gen_labels = gen_outputs[:, prompt_length:].contiguous()
            del logits
            
            gen_token_log_probs = -F.cross_entropy(
                gen_logits.view(-1, gen_logits.size(-1)), 
                gen_labels.view(-1), 
                reduction='none'
            ).view_as(gen_labels)
            
            mask = (gen_labels != tokenizer_handler.tokenizer.pad_token_id).float()
            seq_log_probs = (gen_token_log_probs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
            
            # MEMORY OPTIMIZATION 4: Gradient Accumulation scaling
            loss = -(seq_log_probs * advantages).mean()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Only step the optimizer after accumulating physical batches
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                avg_reward = rewards_tensor.mean().item()
                # Multiply loss back by grad_steps just for accurate TensorBoard logging
                writer.add_scalar("RL/Loss", loss.item() * args.gradient_accumulation_steps, global_step)
                writer.add_scalar("RL/Mean_Reward", avg_reward, global_step)

                if global_step % 10 == 0:
                    logger.info(f"Step {global_step} | Mean Reward: {avg_reward:.4f} | Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")
                    
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_data = {"v": global_v.data if args.n_tie > 1 else {n: p.data for n, p in model.named_parameters() if "v" in n}}
                    torch.save(save_data, os.path.join(checkpoint_dir, "best_model.pt"))

    writer.close()
    logger.info("RL Training complete.")

if __name__ == "__main__":
    main()