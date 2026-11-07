import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, Dataset
import numpy as np
from typing import Sequence
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import pysam


def normalize_chrom(chrom):
    """Remove 'chr' prefix from chromosome name if present."""
    if isinstance(chrom, str) and chrom.startswith('chr'):
        return chrom[3:]
    return chrom


def read_vcf_fine_mapped_enformer(path):
    """Read VCF file from fine-mapped Enformer GTEx variants."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        dtype={0: str},
    ).rename(columns={0: "chrom", 1: "pos", 2: "id"})

    # Parse ref and alt alleles from the 'id' column
    df[['ref', 'alt']] = df['id'].str.split('_', expand=True)[[2, 3]]

    return df

class DNAOneHotEncoder:
    ## Ultrafast one-hot encoder for big DNA sequences
    def __init__(self):
        self.ascii_to_onehot_idx = np.full(128, -1, dtype=np.int8)
        self.ascii_to_onehot_idx[ord('A')] = 0
        self.ascii_to_onehot_idx[ord('C')] = 1
        self.ascii_to_onehot_idx[ord('G')] = 2
        self.ascii_to_onehot_idx[ord('T')] = 3

    def encode(self, seq: str) -> np.ndarray:
        arr = np.zeros((len(seq), 4), dtype=np.int8)
        seq_bytes = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
        idxs = self.ascii_to_onehot_idx[seq_bytes]
        mask = idxs >= 0
        arr[np.arange(len(seq))[mask], idxs[mask]] = 1
        return arr

    def encode_batch(self, seqs: Sequence[str]) -> np.ndarray:
        max_len = max(len(s) for s in seqs)
        out = np.zeros((len(seqs), max_len, 4), dtype=np.int8)

        for i, seq in enumerate(seqs):
            out[i, :len(seq)] = self.encode(seq)

        return out




def load_cagi(path):
    """
    Load CAGI challenge/release TSV files.

    CAGI files have:
    - Lines starting with '##' are metadata comments
    - Line starting with '#' is the header
    - Data rows follow

    Returns:
        DataFrame with columns: chrom, pos, ref, alt, value, confidence
    """
    import gzip
    opener = gzip.open if path.endswith('.gz') else open

    # Count lines starting with ##
    with opener(path, 'rt') as f:
        skip_rows = sum(1 for line in f if line.startswith('##'))

    df = pd.read_csv(path, sep="\t", skiprows=skip_rows)

    # Normalize column names: lowercase and strip '#' prefix
    df.columns = df.columns.str.lower().str.lstrip('#')
    df.chrom = df.chrom.astype(str)

    return df


def load_table(path):
    """
    Load tabular data from various file formats.
    For CSV/TSV/Parquet files, validates that required columns exist.

    Required columns: chrom, pos, ref, alt

    Raises:
        ValueError: If required columns are missing from the file
    """
    # Define required columns for variant data
    REQUIRED_COLS = {"chrom", "pos", "ref", "alt"}

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif "csv" in path:
        df = pd.read_csv(path)
    elif "tsv" in path or path.endswith(".gz"):
        df = pd.read_csv(path, sep="\t")
    elif "vcf" in path:
        # VCF files don't have headers by default, so we manually create column names
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 3, 4],
            dtype={0: str},
        ).rename(columns={0: "chrom", 1: "pos", 3: "ref", 4: "alt"})
    elif "gtf" in path or "gff" in path:
        # GTF/GFF files have predefined format, no validation needed
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            dtype={"chrom": str},
            names=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        df.start -= 1
    else:
        raise ValueError(
            f"Unsupported file format: {path}. "
            "Supported formats: .parquet, .csv, .tsv, .vcf, .gtf, .gff"
        )

    # Normalize column names: lowercase and strip '#' prefix
    df.columns = df.columns.str.lower().str.lstrip('#')

    # Validate required columns (after normalization)
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {path}: {missing_cols}. "
            f"Required: {REQUIRED_COLS}, Found: {set(df.columns)}"
        )

    # Ensure chrom is string type
    df.chrom = df.chrom.astype(str)
    return df


def validate_variants(df, genome):
    """
    Validate variant reference alleles against the genome (vectorized).

    Returns:
        valid_df: DataFrame with variants that match the genome
        mismatched_df: DataFrame with variants that don't match
    """
    df = df.copy()
    df['chrom'] = df['chrom'].apply(normalize_chrom)

    # Vectorized: get genome base for each variant
    def get_genome_base(row):
        try:
            return genome._genome[row['chrom']][row['pos'] - 1].upper()
        except (KeyError, IndexError):
            return None

    df['genome_base'] = df.apply(get_genome_base, axis=1)
    df['ref_upper'] = df['ref'].str.upper()

    mask = df['genome_base'] == df['ref_upper']
    valid_df = df[mask].drop(columns=['genome_base', 'ref_upper']).reset_index(drop=True)
    mismatched_df = df[~mask].drop(columns=['ref_upper']).reset_index(drop=True)

    return valid_df, mismatched_df


class Genome:
    ## verifed with blast genome.get_seq_fwd_rev('10', 500_000, 500_000+1000)
    def __init__(self, path):
        self._genome = self._load_fasta(path)
        self.chrom_sizes = {chrom: len(seq) for chrom, seq in self._genome.items()}

    @staticmethod
    def _load_fasta(path):
        """
        Load FASTA file using pysam and return as dict.
        """
        fasta = pysam.FastaFile(path)
        genome = {}
        try:
            for chrom in fasta.references:
                genome[normalize_chrom(chrom)] = fasta.fetch(chrom)
        finally:
            fasta.close()

        return genome

    @staticmethod
    def _reverse_complement(seq):
        """Reverse complement a DNA sequence."""
        complement_table = str.maketrans('ACGTacgt', 'TGCAtgca')
        return seq.translate(complement_table)[::-1]

    def get_seq(self, chrom, start, end, strand="+"):
        # Normalize chromosome name
        chrom = normalize_chrom(chrom)

        if chrom not in self.chrom_sizes:
            raise KeyError(f"Chromosome '{chrom}' not found in genome. Available: {list(self.chrom_sizes.keys())[:10]}...")

        chrom_size = self.chrom_sizes[chrom]
        seq = self._genome[chrom][max(start, 0):min(end, chrom_size)]
        if start < 0:
            seq = "N" * (-start) + seq  # left padding
        if end > chrom_size:
            seq = seq + "N" * (end - chrom_size)  # right padding
        if strand == "-":
            seq = self._reverse_complement(seq)
        return seq

    def get_seq_fwd_rev(self, chrom, start, end):
        seq_fwd = self.get_seq(chrom, start, end)
        seq_rev = self._reverse_complement(seq_fwd)
        return seq_fwd, seq_rev


class VariantDataset(TorchDataset):  # Inherit from PyTorch Dataset
    """
    PyTorch Dataset for variant impact prediction using seq2fun models.

    Generates reference and alternate sequences in both forward and reverse
    orientations for each variant, suitable for reverse-complement aware models.

    Args:
        data: DataFrame or path to file with columns:
            - chrom: Chromosome name (str)
            - pos: Variant position (int, 1-based)
            - ref: Reference allele (str)
            - alt: Alternate allele (str)
        genome: Genome object for sequence extraction
        seq_len: Total sequence length to extract (default: 1000)
    """

    def __init__(self, data, genome, seq_len=1000):
        # Handle both DataFrame and file path inputs
        if isinstance(data, str):
            self.df = load_table(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data must be a pandas DataFrame or a file path string")

        self.genome = genome
        self.oh = DNAOneHotEncoder()
        self.seq_len = seq_len
        self.half_len = seq_len // 2

        # Normalize chromosome names
        self.df['chrom'] = self.df['chrom'].apply(normalize_chrom)
        # Convert to 0-based position for easier indexing
        self.df['pos_0'] = self.df['pos'] - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get one-hot encoded sequences for a variant.

        Returns:
            dict with keys:
                - x_ref_fwd: Reference sequence, forward strand (seq_len, 4)
                - x_alt_fwd: Alternate sequence, forward strand (seq_len, 4)
                - x_ref_rev: Reference sequence, reverse strand (seq_len, 4)
                - x_alt_rev: Alternate sequence, reverse strand (seq_len, 4)
        """
        row = self.df.iloc[idx]

        assert len(row["ref"]) == len(row["alt"]) == 1, "Only SNPs are supported in this VEP implementation."

        # Get reference sequences centered on variant position
        fw_seq, rv_seq = self.genome.get_seq_fwd_rev(
            row["chrom"],
            row["pos_0"] - self.half_len,
            row["pos_0"] + self.half_len
        )

        # Variant positions in the extracted sequence
        pos_fwd = self.half_len
        pos_rev = pos_fwd - 1 if self.seq_len % 2 == 0 else pos_fwd

        # Create mutated forward sequence
        alt_fw_seq = (
            fw_seq[:pos_fwd]
            + row["alt"]
            + fw_seq[pos_fwd + 1:]
        )

        # Create mutated reverse sequence
        alt_rc = self.genome._reverse_complement(row["alt"])
        alt_rv_seq = (
            rv_seq[:pos_rev]
            + alt_rc
            + rv_seq[pos_rev + 1:]
        )

        # Double checks
        assert fw_seq[pos_fwd] == row["ref"], f"Reference allele mismatch in forward strand. Idx {idx} Expected {row['ref']}, got {fw_seq[pos_fwd]}"
        assert rv_seq[pos_rev] == self.genome._reverse_complement(row["ref"]), "Reference allele mismatch in reverse strand."

        # One-hot encode all sequences
        x_ref_fwd = torch.from_numpy(self.oh.encode(fw_seq)).float()
        x_alt_fwd = torch.from_numpy(self.oh.encode(alt_fw_seq)).float()
        x_ref_rev = torch.from_numpy(self.oh.encode(rv_seq)).float()
        x_alt_rev = torch.from_numpy(self.oh.encode(alt_rv_seq)).float()

        return {
            'x_ref_fwd': x_ref_fwd,
            'x_alt_fwd': x_alt_fwd,
            'x_ref_rev': x_ref_rev,
            'x_alt_rev': x_alt_rev,
        }

    def collate_fn(self, batch):
        """
        Collate function for DataLoader to batch sequences.

        Args:
            batch: List of dicts from __getitem__

        Returns:
            dict with batched tensors of shape (batch_size, seq_len, 4)
        """
        x_ref_fwd = torch.stack([item['x_ref_fwd'] for item in batch])
        x_alt_fwd = torch.stack([item['x_alt_fwd'] for item in batch])
        x_ref_rev = torch.stack([item['x_ref_rev'] for item in batch])
        x_alt_rev = torch.stack([item['x_alt_rev'] for item in batch])

        return {
            'x_ref_fwd': x_ref_fwd,
            'x_alt_fwd': x_alt_fwd,
            'x_ref_rev': x_ref_rev,
            'x_alt_rev': x_alt_rev,
        }
