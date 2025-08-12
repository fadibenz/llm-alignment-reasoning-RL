from pathlib import Path
import json
import random
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        assert len(data_tensor) == len(labels_tensor), "Data and labels must match in length"
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_validation_data(input_path: Path, prompt_template: str):
    prompts, answers = [], []

    if input_path.suffix != ".jsonl":
        raise ValueError("Input file must be in .jsonl format.")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json_obj = json.loads(line)
            if "{question}" not in prompt_template:
                raise ValueError("Prompt template must contain '{question}'.")
            prompt = prompt_template.replace("{question}", json_obj["problem"])
            prompts.append(prompt)
            answers.append(json_obj["answer"])

    return prompts, answers

def load_training_data(input_path, number_samples):
    prompts, answers = [], []

    if input_path.suffix != ".jsonl":
        raise ValueError("Input file must be in .jsonl format.")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json_obj = json.loads(line)
            prompts.append(json_obj["prompt"])
            answers.append(json_obj["response"])

    combined = list(zip(prompts, answers))
    random.shuffle(combined)
    prompts, answers = zip(*combined)

    prompts = list(prompts)[:number_samples]
    answers = list(answers)[:number_samples]

    return prompts, answers
