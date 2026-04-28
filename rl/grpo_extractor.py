import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .reward_fns import gsm8k_exact_match_reward

def accumulate_rl_fisher_gradients(
    model: nn.Module, 
    tokenizer, 
    dataloader, 
    target_layer_names: list, 
    num_generations: int = 4, 
    device="cuda"
):
    model.eval()
    fisher_dict = {name: torch.zeros_like(model.get_submodule(name).weight, device=device) 
                   for name in target_layer_names}
    
    for name, param in model.named_parameters():
        if any(target in name for target in target_layer_names):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for batch in tqdm(dataloader, desc="Extracting RL Fisher Gradients"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ground_truths = batch["completion"]
        
        # Calculate actual prompt length ignoring padding
        actual_max_len = attention_mask.sum(dim=1).max().item()
        
        # Guardrail: Token Cap
        if actual_max_len > 600:
            continue
            
        # Because padding_side="left", we slice off the useless padding 
        # to massively speed up the generation phase
        pad_len = input_ids.shape[1] - actual_max_len
        sliced_input_ids = input_ids[:, pad_len:]
        sliced_attention_mask = attention_mask[:, pad_len:]
        
        model.zero_grad()
        
        # 1. Generate using the SLICED inputs
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=sliced_input_ids,
                attention_mask=sliced_attention_mask,
                max_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=num_generations,
                pad_token_id=tokenizer.pad_token_id
            )
        
        gen_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
        expanded_truths = [t for t in ground_truths for _ in range(num_generations)]
        rewards = gsm8k_exact_match_reward(gen_texts, expanded_truths)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        advantages = torch.zeros_like(rewards_tensor)
        for i in range(0, len(rewards_tensor), num_generations):
            group_rewards = rewards_tensor[i:i+num_generations]
            adv = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            advantages[i:i+num_generations] = adv

        model.train() 
        
        outputs = model(
            input_ids=gen_outputs, 
            attention_mask=(gen_outputs != tokenizer.pad_token_id).long()
        )
        logits = outputs.logits
        del outputs 
        
        # Calculate prompt length based on the SLICED sequence length
        prompt_length = sliced_input_ids.shape[1]
        
        gen_logits = logits[:, prompt_length - 1 : -1, :].contiguous()
        gen_labels = gen_outputs[:, prompt_length:].contiguous()
        del logits 
        
        gen_token_log_probs = -F.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)), 
            gen_labels.view(-1), 
            reduction='none'
        ).view_as(gen_labels)
        
        mask = (gen_labels != tokenizer.pad_token_id).float()
        seq_log_probs = (gen_token_log_probs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        
        loss = -(seq_log_probs * advantages).mean()
        loss.backward()
        
        model.eval()
        
        with torch.no_grad():
            for name in target_layer_names:
                weight_param = model.get_submodule(name).weight
                if weight_param.grad is not None:
                    fisher_dict[name] += (weight_param.grad ** 2) / len(dataloader)
                    
    return fisher_dict