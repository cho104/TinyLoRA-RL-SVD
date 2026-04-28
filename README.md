# Parameter-Efficient LLM Fine-Tuning with TinyLoRA & Improved SVD via Reinforcement Learning Alignment

## Summary
This project explores the efficient tuning of Large Language Models (LLMs) using TinyLoRA, a method that demonstrates improved mathematical reasoning capabilities using only 13 trainable parameters and random projection matrices. 

Standard Singular Value Decomposition (SVD) has a limitation in that it ignores the data distribution of downstream tasks. To overcome this, this project applies Activation-aware SVD (AW-SVD) reflecting activation magnitudes, and Fisher-Weighted SVD (F-SVD) reflecting Fisher information. Experimental results show that the RL-F-SVD initialization technique, which directly reflects gradient sensitivity toward the target reward, overcomes the superficial pattern overfitting of AW-SVD and demonstrates excellent generalization performance (Pass@1 79%).

## Key Features
* **TinyLoRA (13 Parameters)**: Leverages extreme parameter efficiency to steer models by tapping into pre-existing "Neural Thickets" around pretrained weights.
* **Activation-aware SVD (RL-AW-SVD)**: Transforms the activation magnitude of each input channel during RL exploration into a diagonal matrix, multiplying it by the weights before decomposition.
* **Fisher-Weighted SVD (RL-F-SVD)**: Calculates a diagonal Fisher information matrix based on loss gradients reflecting RL Advantages, using it to perform SVD.
* **Skew-Symmetric Structural Constraints**: Forces the adapter's update matrix into a skew-symmetric format (AW = -AWT) to act as a rotational operation in extreme parameter sharing. *(Note: Experiments revealed this heavily suppresses model expressiveness, leading to underfitting).*
* **RL-Calibrated SFT Paradigm Transfer**: Injects the optimal bottleneck discovered via RL (RL-F-SVD) into low-cost SFT training, functioning as a regularizer against noisy SFT feedback.

## Model & Dataset
* **Base Model**: `Qwen2.5-3B-Instruct`. Loaded with bfloat16 precision and sdpa to optimize VRAM usage.
* **Dataset**: `openai/GSM8K` (Primary school-level math problems).
    * **Calibration**: 1000 samples extracted from the training set with 4 rollouts for RL-AWQ and RL-F-SVD calculation.
    * **Training**: 7.47k samples for GRPO and SFT fine-tuning.
    * **Evaluation**: 100 samples from the test set for fast evaluation.

## Repository Structure
```text
peft_project/
├── core/                   # Core model logic (TinyLoRA, Projection, etc.)
├── data/                   # Data loading and tokenization
├── initialization/         # SVD and AWQ initialization modules 
│   └── rl/                 # RL-based initializations (awq_svd.py, base_svd.py, fisher_svd.py)
├── rl/                     # Reinforcement learning logic (GRPO extractor, Reward functions)
├── utils/                  # Common utilities (arguments, logger, metrics)
├── scripts/                # Execution scripts (Training, Evaluation, Extraction)
│   ├── train_rl.py         # Main RL training script
│   ├── calibrate_svd.py    # Calibration script
│   └── evaluate_*.py       # Evaluation scripts
└── outputs/                # Experimental results
    ├── checkpoints/        # Model checkpoints
    ├── svd_cache/          # SVD cache data
    └── eval_results/       # Qualitative evaluation results (*.md)
```

## Key Results

| Initialization Method | Best Pass@1 | Final Mean Reward |
| :--- | :--- | :--- |
| **Baseline SVD** | 77% | 0.7545 |
| **RL-AW-SVD (ours)** | 72% | 0.7989 |
| **RL-F-SVD (ours)** | **79%** | **0.8103** |

* **Quantitative Analysis**: RL-AW-SVD initially secures rewards quickly, but RL-F-SVD surpasses it around 1200 steps to reach a higher performance ceiling. RL-AW-SVD suffered from overfitting (high training reward but low test accuracy), while RL-F-SVD successfully generalized.
* **Qualitative Analysis**: In an extremely restricted 13-parameter budget, SFT-based models wasted capacity memorizing dataset formatting (e.g., text tags, punctuation), leading to meaningless symbol repetition. In contrast, RL-based models allocated their budget purely to logical computation and successfully completed Chain-of-Thought (CoT) reasoning without representation collapse.

## Hardware & Optimizations
* **Environment**: VSCode via SSH, Miniconda.
* **Hardware**: NVIDIA RTX A5000.
* **Optimizations**: Gradient Checkpointing and Fused Cross-Entropy (Log-Softmax optimization) were applied to resolve VRAM shortages during multi-rollout RL training.

## References
* [1] Gan, Y., & Isola, P. (2026). Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights.
* [2] Morris, J. X., Mireshghallah, N., Ibrahim, M., & Mahloujifar, S. (2026). Learning to Reason in 13 Parameters.
* [3] Yuan, Z., et al. (2023). Asvd: Activation-aware singular value decomposition for compressing large language models.