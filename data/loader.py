from datasets import load_dataset

def load_math_datasets(dataset_name: str = "gsm8k", split: str = "train"):
    """
    Loads the requested math reasoning dataset.
    Supported: 'gsm8k', 'math500'
    """
    if dataset_name.lower() == "gsm8k":
        # GSM8K has 'main' and 'socratic' subsets. 'main' is standard.
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Standardize column names to 'question' and 'answer'
        dataset = dataset.rename_column("question", "prompt")
        dataset = dataset.rename_column("answer", "completion")
        return dataset
        
    elif dataset_name.lower() == "math500":
        # MATH500 is a high-quality subset of the MATH dataset
        dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
        dataset = dataset.rename_column("problem", "prompt")
        dataset = dataset.rename_column("solution", "completion")
        return dataset
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")