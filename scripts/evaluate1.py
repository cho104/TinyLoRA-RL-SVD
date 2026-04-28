import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from utils.arguments import parse_args
from utils.logger import setup_logger
from data.loader import load_math_datasets
from data.tokenizer import QwenMathTokenizer
from rl.reward_fns import gsm8k_exact_match_reward
from scripts.train import replace_with_tinylora

def main():
    args = parse_args()
    logger = setup_logger("Evaluate")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    svd_components = torch.load(args.svd_init_path, map_location="cpu")
    model, global_v = replace_with_tinylora(model, ["down_proj"], svd_components, args)
    
    # Assuming run dir corresponds to outputs/checkpoints/YOUR_RUN_DIR (passed manually typically)
    run_dir = "outputs/checkpoints/RL_none_u13_tie40_1775762985"

    trained_weights = torch.load(f"{run_dir}/best_model.pt", map_location="cpu")

    # FIX: Properly standardize weight payload fetching 
    if args.n_tie > 1:
        global_v.data.copy_(trained_weights["v"].data)
    else:
        for n, p in model.named_parameters():
            if "v" in n and n in trained_weights["v"]:
                p.data.copy_(trained_weights["v"][n].data)
                
    model = model.to(device)
    model.eval()

    tokenizer_handler = QwenMathTokenizer(args.model_name)
    val_dataset = load_math_datasets(args.dataset, split="test[:100]")
    
    correct = 0
    total = len(val_dataset)
    
    logger.info(f"Starting Evaluation on {total} samples...")
    with torch.no_grad():
        for i in tqdm(range(total)):
            prompt = val_dataset[i]["prompt"]
            ground_truth = val_dataset[i]["completion"]
            
            formatted_prompt = tokenizer_handler.apply_chat_template(prompt)
            inputs = tokenizer_handler.tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer_handler.tokenizer.pad_token_id,
                do_sample=False
            )
            
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            generated_text = tokenizer_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            reward = gsm8k_exact_match_reward([generated_text], [ground_truth])[0]
            correct += int(reward)
            
    accuracy = (correct / total) * 100
    logger.info(f"Evaluation Complete. Pass@1 Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()