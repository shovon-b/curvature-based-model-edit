import torch
import os
from datasets import load_dataset, Dataset, load_from_disk

from torch.utils.data import Dataset as TorchDataset


class SimpleTextDataset(TorchDataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            print(f"Data path {data_path} not found. Running civil_data processing...")
            from data import civil_data
            processor = civil_data()
            processor.process()
        self.ds = load_from_disk(data_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Access the internal dataset using the new name
        return self.ds[idx]['text']

class civil_data:
    def __init__(self, toxicity_threshold=0.8, non_toxicity_threshold=0.0, d_split="validation"):
        self.tox = toxicity_threshold
        self.ntox = non_toxicity_threshold
        self.out_dir = "data"
        self.d_split = d_split

    def process(self):
        print(f"Loading Civil Comments dataset...")
        dataset = load_dataset("google/civil_comments", split=self.d_split)
        toxic = dataset.filter(lambda x: x['toxicity'] >= self.tox)
        nontoxic = dataset.filter(lambda x: x['toxicity'] <= self.ntox)
        length = min(len(toxic), len(nontoxic))
        print(f"dataset size={length}")
        toxic = toxic.shuffle(seed=2025).select(range(length))
        nontox = nontoxic.shuffle(seed=2025).select(range(length))
        tox_path = os.path.join(self.out_dir, f"toxic_train")
        nontox_path = os.path.join(self.out_dir, f"nontoxic_train")
        toxic.save_to_disk(tox_path)
        nontox.save_to_disk(nontox_path)
        print(f"Data saved to local directory: './{self.out_dir}'")
