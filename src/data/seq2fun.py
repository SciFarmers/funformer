
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Literal
from tqdm import tqdm
from src.bio.genome import Genome, Tokenizer
import logging
from datasets import load_from_disk



class Seq2FunDataset(Dataset):
    """
    split: train | validation | test
    data_dir: path to HF dataset
    genome_path: path to hg38.fa
    sequence_length: length of input sequence (default: 1000)
    max_samples: if set, limits the number of samples loaded, useful for debugging
    """
    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        data_dir: str,
        genome_path: str,
        sequence_length: int = 1000,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.genome_path = genome_path
        self.sequence_length = sequence_length

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found at: {data_dir}")

        self.data_indices = load_from_disk(data_dir)[split]
        self.half_len = sequence_length // 2
        self.max_samples = max_samples
        self.tokenizer = Tokenizer()
        self.genome = None  # Will be initialized in worker processes



    def _ensure_genome(self):
        if self.genome is None:
            self.genome = Genome(self.genome_path, self.tokenizer)

    def __getitem__(self, idx):
        self._ensure_genome()
        data_point = self.data_indices[idx]
        chrom = data_point["chrom"]
        start = data_point["start"]
        end = data_point["end"]
        labels = data_point["labels"]

        ## get sequence window
        center = (start + end) // 2
        seq_start = center - self.half_len
        seq_end = center + self.half_len 

        fwd_tokens, rev_tokens = self.genome.get_seq_window(chrom, seq_start, seq_end)

        fwd_tokens = torch.from_numpy(fwd_tokens).long()
        rev_tokens = torch.from_numpy(rev_tokens).long()

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32) if isinstance(labels, (list, np.ndarray)) else torch.tensor([labels], dtype=torch.float32)
        else:
            labels = labels.float()

        return {
            "fwd_ids": fwd_tokens,
            "rev_ids": rev_tokens,
            "labels": labels,
        }

    def __len__(self):
        if self.max_samples is None:
            return len(self.data_indices)
        return min(self.max_samples, len(self.data_indices))


def test_multi_cpu_loader():
    dataset = Seq2FunDataset(
        split="train",
        data_dir="/Users/thanhdng/github/data_factory/seq2fun_binary/datasets/hf_ENTEX_423_beds",
        genome_path="/Users/thanhdng/Documents/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        sequence_length=1000,
    )
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=6, shuffle=True)
    for batch in tqdm(dataloader, desc="Testing multi-CPU DataLoader"):
        assert "fwd_ids" in batch and "rev_ids" in batch and "labels" in batch
        assert batch["fwd_ids"].shape[1] == 1000
        assert batch["rev_ids"].shape[1] == 1000
        assert batch["labels"].shape[0] == batch["fwd_ids"].shape[0]
    print("Multi-CPU DataLoader test passed!")
    exit(0)

if __name__ == "__main__":
    test_multi_cpu_loader()