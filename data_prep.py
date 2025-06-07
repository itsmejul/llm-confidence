from datasets import load_dataset

def prepare_dataset_for_correlation_analysis(dataset_name, n_samples):
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", split="test", name="main")
        dataset = dataset.remove_columns("answer")
        dataset = dataset.rename_column("question", "prompt")
    elif dataset_name == "writingprompts":
        dataset = load_dataset("euclaise/writingprompts", split="validation")
        dataset = dataset.remove_columns("story")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    return dataset