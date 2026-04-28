import os
import json
import torch
from transformers import AutoModelForCausalLM

from utils.arguments import parse_args
from utils.logger import setup_logger
from data.loader import load_math_datasets
from data.tokenizer import QwenMathTokenizer
from scripts.train import replace_with_tinylora

def main():
    args = parse_args()
    logger = setup_logger("QualitativeExtraction")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 처리할 모델 체크포인트 폴더 경로들을 리스트로 작성합니다.
    run_dirs = [
        "outputs/checkpoints/RL_none_u13_tie40_1775762985",
    ]

    # 2. 무거운 베이스 모델과 토크나이저는 단 한 번만 로드합니다.
    logger.info(f"Loading Base Model: {args.model_name} (This happens only once)")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    svd_components = torch.load(args.svd_init_path, map_location="cpu")
    model, global_v = replace_with_tinylora(model, ["down_proj"], svd_components, args)
    model = model.to(device)
    
    tokenizer_handler = QwenMathTokenizer(args.model_name)
    val_dataset = load_math_datasets(args.dataset, split="test")
    sample_indices = [0, 10, 50, 100, 200]

    # 3. 각 체크포인트를 순회하며 가중치를 바꿔끼우고 추론을 진행합니다.
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir.rstrip('/'))
        logger.info(f"\n[{run_name}] 모델 평가 시작...")
        
        weight_path = f"{run_dir}/best_model.pt"
        if not os.path.exists(weight_path):
            logger.warning(f"체크포인트를 찾을 수 없습니다: {weight_path}. 건너뜁니다.")
            continue

        # 가중치 로드 및 바꿔끼우기 (In-place 업데이트)
        trained_weights = torch.load(weight_path, map_location="cpu")
        if args.n_tie > 1:
            global_v.data.copy_(trained_weights["v"].data)
        else:
            for n, p in model.named_parameters():
                if "v" in n and n in trained_weights["v"]:
                    p.data.copy_(trained_weights["v"][n].data)
                    
        model.eval()
        output_data = []
        
        for idx in sample_indices:
            prompt = val_dataset[idx]["prompt"]
            ground_truth = val_dataset[idx]["completion"]
            
            formatted_prompt = tokenizer_handler.apply_chat_template(prompt)
            inputs = tokenizer_handler.tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            generated_text = tokenizer_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            output_data.append({
                "Prompt": prompt,
                "Ground Truth": ground_truth,
                "Generated Output": generated_text
            })
            
        # 4. 파일명이 덮어씌워지지 않도록 체크포인트 이름(run_name)을 포함하여 저장합니다.
        out_file = f"outputs/qualitative_{run_name}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# Qualitative Outputs for Run: {run_name}\n\n")
            for i, data in enumerate(output_data):
                f.write(f"### Example {i+1}\n")
                f.write(f"**Prompt:** {data['Prompt']}\n\n")
                f.write(f"**Ground Truth:**\n```\n{data['Ground Truth']}\n```\n\n")
                f.write(f"**Model Output:**\n```\n{data['Generated Output']}\n```\n\n")
                f.write("---\n")
                
        logger.info(f"[{run_name}] saved: {out_file}")

if __name__ == "__main__":
    main()