import torch
from transformers import AutoTokenizer

class QwenMathTokenizer:
    """
    Handles tokenization and chat template formatting specific to Qwen2.5.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        
        # Qwen models often don't have a default pad token, use eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_chat_template(self, prompt: str, completion: str = None) -> str:
        """
        Wraps the prompt in Qwen's <|im_start|> and <|im_end|> tags.
        """
        messages = [
            {"role": "system", "content": "You are a helpful mathematical reasoning assistant. Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": prompt}
        ]
        
        if completion:
            messages.append({"role": "assistant", "content": completion})
            
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not completion)
    
    def tokenize_rl_dataset(self, dataset):
        """
        Tokenizes only the prompt for RL generation, omitting the SFT completion.
        """
        def format_and_tokenize(example):
            # NO COMPLETION PROVIDED
            prompt_text = self.apply_chat_template(example["prompt"]) 
            tokenized = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length", # We pad to max_length to keep dataloaders happy
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "completion": example["completion"] # Keep raw string for reward scoring
            }

        return dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

    def tokenize_dataset(self, dataset):
        """
        Maps the chat template across the HuggingFace dataset and applies SFT masking.
        """
        def format_and_tokenize(example):
            # 1. Tokenize full text (prompt + completion)
            full_text = self.apply_chat_template(example["prompt"], example["completion"])
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 2. Initialize labels
            labels = tokenized["input_ids"].clone()
            
            # 3. Calculate exact prompt length to mask out the question
            prompt_text = self.apply_chat_template(example["prompt"]) 
            # Force add_special_tokens=False because the chat_template already includes them
            prompt_tokenized = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            prompt_length = prompt_tokenized["input_ids"].shape[1]
            
            # 4. Mask out the prompt (loss only on the assistant's completion)
            # Guard against edge cases where prompt > max_length
            mask_len = min(prompt_length, self.max_length)
            labels[0, :mask_len] = -100 
            
            # 5. Mask out padding tokens
            labels[tokenized["attention_mask"] == 0] = -100
            
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": labels[0]
            }

        return dataset.map(format_and_tokenize, remove_columns=dataset.column_names)