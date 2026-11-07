"""
Self-contained utilities for VEP and evaluation in src_evaluate.

Mirrors src/seq2fun_vep/utils.py — genome access, one-hot encoding, and
VariantDataset are all implemented here without importing from src.bio.
"""

import numpy as np
import pandas as pd
from typing import Sequence
from tqdm import tqdm

import pysam
import torch
from torch.utils.data import Dataset as TorchDataset

tqdm.pandas()


# ---------------------------------------------------------------------------
# Chromosome normalisation
# ---------------------------------------------------------------------------

def normalize_chrom(chrom):
    """Strip 'chr' prefix from chromosome name if present."""
    if isinstance(chrom, str) and chrom.startswith("chr"):
        return chrom[3:]
    return chrom


# ---------------------------------------------------------------------------
# VCF reader — enformer fine-mapped GTEx format
# ---------------------------------------------------------------------------

def read_vcf_fine_mapped_enformer(path: str) -> pd.DataFrame:
    """
    Read enformer-style fine-mapped VCF.

    Columns read: 0=chrom, 1=pos (1-based), 2=id (CHROM_POS_REF_ALT_b38).
    Ref and alt are parsed from the id field.
    Only SNPs (single-base substitutions) are retained.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        dtype={0: str},
    ).rename(columns={0: "chrom", 1: "pos", 2: "id"})

    df[["ref", "alt"]] = df["id"].str.split("_", expand=True)[[2, 3]]
    df = df.drop(columns=["id"])

    # Keep only SNPs
    snp_mask = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    n_skipped = (~snp_mask).sum()
    if n_skipped:
        print(f"  Skipping {n_skipped} non-SNP variants")
    df = df[snp_mask].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# DNA one-hot encoder
# ---------------------------------------------------------------------------

class DNAOneHotEncoder:
    """Ultrafast one-hot encoder for DNA sequences using ASCII lookup table."""

    def __init__(self):
        # Channel order matches the training tokenizer (tokens_to_onehot in trainer.py):
        #   Vocab: N=0, A=1, T=2, G=3, C=4  →  one-hot: A/N→ch0, T→ch1, G→ch2, C→ch3
        self.ascii_to_onehot_idx = np.full(128, -1, dtype=np.int8)
        self.ascii_to_onehot_idx[ord("A")] = 0
        self.ascii_to_onehot_idx[ord("T")] = 1
        self.ascii_to_onehot_idx[ord("G")] = 2
        self.ascii_to_onehot_idx[ord("C")] = 3

    def encode(self, seq: str) -> np.ndarray:
        """Encode a DNA string → (len, 4) int8 one-hot array."""
        arr = np.zeros((len(seq), 4), dtype=np.int8)
        seq_bytes = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
        idxs = self.ascii_to_onehot_idx[seq_bytes]
        mask = idxs >= 0
        arr[np.arange(len(seq))[mask], idxs[mask]] = 1
        return arr

    def encode_batch(self, seqs: Sequence[str]) -> np.ndarray:
        """Encode a list of DNA strings → (N, max_len, 4) int8 array."""
        max_len = max(len(s) for s in seqs)
        out = np.zeros((len(seqs), max_len, 4), dtype=np.int8)
        for i, seq in enumerate(seqs):
            out[i, : len(seq)] = self.encode(seq)
        return out


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

class Genome:
    """
    Load a FASTA genome into RAM via pysam and provide sequence extraction
    with N-padding at boundaries and reverse-complement support.
    """

    def __init__(self, path: str):
        self._genome = self._load_fasta(path)
        self.chrom_sizes = {chrom: len(seq) for chrom, seq in self._genome.items()}

    @staticmethod
    def _load_fasta(path: str) -> dict:
        fasta = pysam.FastaFile(path)
        genome: dict = {}
        try:
            for chrom in fasta.references:
                genome[normalize_chrom(chrom)] = fasta.fetch(chrom)
        finally:
            fasta.close()
        return genome

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        complement_table = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(complement_table)[::-1]

    def get_seq(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        """
        Extract a sequence window [start, end) with N-padding at boundaries.

        Parameters
        ----------
        chrom : str  (with or without 'chr' prefix)
        start : int  0-based, inclusive
        end   : int  0-based, exclusive
        strand: '+' or '-'
        """
        chrom = normalize_chrom(chrom)
        if chrom not in self.chrom_sizes:
            raise KeyError(
                f"Chromosome '{chrom}' not found in genome. "
                f"Available (first 10): {list(self.chrom_sizes)[:10]}"
            )
        chrom_size = self.chrom_sizes[chrom]
        seq = self._genome[chrom][max(start, 0) : min(end, chrom_size)]
        if start < 0:
            seq = "N" * (-start) + seq
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)
        if strand == "-":
            seq = self._reverse_complement(seq)
        return seq

    def get_seq_fwd_rev(self, chrom: str, start: int, end: int):
        """Return (forward_seq, reverse_complement_seq) for a window."""
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = self._reverse_complement(seq_fwd)
        return seq_fwd, seq_rev


# ---------------------------------------------------------------------------
# VariantDataset
# ---------------------------------------------------------------------------

class VariantDataset(TorchDataset):
    """
    PyTorch Dataset for VEP.

    Each item contains four one-hot float tensors of shape (seq_len, 4):
        x_ref_fwd, x_alt_fwd, x_ref_rev, x_alt_rev

    Only SNPs (len(ref)==len(alt)==1) are supported.
    """

    def __init__(self, data, genome: Genome, seq_len: int = 1000):
        if isinstance(data, str):
            self.df = read_vcf_fine_mapped_enformer(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data must be a pandas DataFrame or a file path string")

        self.genome = genome
        self.oh = DNAOneHotEncoder()
        self.seq_len = seq_len
        self.half_len = seq_len // 2

        # Normalise chromosomes and convert to 0-based position
        self.df["chrom"] = self.df["chrom"].apply(normalize_chrom)
        self.df["pos_0"] = self.df["pos"] - 1
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        assert len(row["ref"]) == len(row["alt"]) == 1, (
            f"Only SNPs are supported. idx={idx}: ref={row['ref']}, alt={row['alt']}"
        )

        start = row["pos_0"] - self.half_len
        end   = row["pos_0"] + self.half_len

        fw_seq, rv_seq = self.genome.get_seq_fwd_rev(row["chrom"], start, end)

        pos_fwd = self.half_len
        pos_rev = pos_fwd - 1 if self.seq_len % 2 == 0 else pos_fwd

        # Build mutated sequences
        alt_fw_seq = fw_seq[:pos_fwd] + row["alt"] + fw_seq[pos_fwd + 1:]
        alt_rc = self.genome._reverse_complement(row["alt"])
        alt_rv_seq = rv_seq[:pos_rev] + alt_rc + rv_seq[pos_rev + 1:]

        # Sanity checks
        assert fw_seq[pos_fwd].upper() == row["ref"].upper(), (
            f"Ref mismatch (fwd) at idx {idx}: expected {row['ref']}, got {fw_seq[pos_fwd]}"
        )
        assert rv_seq[pos_rev].upper() == self.genome._reverse_complement(row["ref"]).upper(), (
            f"Ref mismatch (rev) at idx {idx}"
        )

        return {
            "x_ref_fwd": torch.from_numpy(self.oh.encode(fw_seq)).float(),
            "x_alt_fwd": torch.from_numpy(self.oh.encode(alt_fw_seq)).float(),
            "x_ref_rev": torch.from_numpy(self.oh.encode(rv_seq)).float(),
            "x_alt_rev": torch.from_numpy(self.oh.encode(alt_rv_seq)).float(),
        }

    def collate_fn(self, batch: list) -> dict:
        return {
            "x_ref_fwd": torch.stack([b["x_ref_fwd"] for b in batch]),
            "x_alt_fwd": torch.stack([b["x_alt_fwd"] for b in batch]),
            "x_ref_rev": torch.stack([b["x_ref_rev"] for b in batch]),
            "x_alt_rev": torch.stack([b["x_alt_rev"] for b in batch]),
        }


# ---------------------------------------------------------------------------
# Variant validation helper
# ---------------------------------------------------------------------------

def validate_variants(df: pd.DataFrame, genome: Genome):
    """
    Validate ref alleles against the genome.

    Returns (valid_df, mismatched_df).
    """
    df = df.copy()
    df["chrom"] = df["chrom"].apply(normalize_chrom)

    def get_genome_base(row):
        try:
            return genome._genome[row["chrom"]][row["pos"] - 1].upper()
        except (KeyError, IndexError):
            return None

    df["genome_base"] = df.apply(get_genome_base, axis=1)
    df["ref_upper"] = df["ref"].str.upper()
    mask = df["genome_base"] == df["ref_upper"]
    valid_df = df[mask].drop(columns=["genome_base", "ref_upper"]).reset_index(drop=True)
    mismatched_df = df[~mask].drop(columns=["ref_upper"]).reset_index(drop=True)
    return valid_df, mismatched_df
