import re

def extract_boxed_answer(text: str) -> str:
    """Extracts the final answer from a \boxed{} LaTeX tag."""
    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""

def gsm8k_exact_match_reward(completions: list[str], ground_truths: list[str]) -> list[float]:
    """
    Evaluates generated completions against ground truth answers.
    Returns 1.0 for a correct exact match, 0.0 otherwise.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truths):
        pred_ans = extract_boxed_answer(comp)
        truth_ans = extract_boxed_answer(truth)
        
        # Fallback if ground truth is just the raw string and not boxed
        if not truth_ans:
            truth_ans = truth.split("####")[-1].strip() if "####" in truth else truth.strip()
            
        if pred_ans and pred_ans == truth_ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards