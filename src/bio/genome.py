import pysam
import numpy as np
import re


class Tokenizer:
    """
    Byte-level DNA tokenizer. Maps A/T/G/C (upper or lower) to indices 0-3;
    any other character (N, ambiguity codes, padding) maps to the pad index.

    Default vocab: "NATGC"  →  N=0 (pad), A=1, T=2, G=3, C=4 (pad).
    """

    def __init__(self, vocab: str = "NATGC"):
        self.vocab = vocab
        self.pad_token = "N"
        self.pad_index = vocab.index(self.pad_token)
        self.table = np.full((256,), self.pad_index, dtype=np.uint8)
        for i, c in enumerate(vocab):
            self.table[ord(c)] = i
            self.table[ord(c.lower())] = i

    def __call__(self, seq: str) -> np.ndarray:
        """Tokenize a DNA string into a uint8 numpy array."""
        return self.table[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

    def decode(self, ids: np.ndarray) -> str:
        """Decode a token array back to a DNA string."""
        return "".join(self.vocab[i] for i in ids)


_MITO_ALIASES = {"M", "MT"}
#_VALID_CHROMS = {str(i) for i in range(1, 99)} | {"X", "Y", "MT"}

_VALID_CHROMS = {str(i) for i in range(1, 99)}


def normalize_chrom(chrom: str) -> str | None:
    """
    Normalize chromosome name to Ensembl format.
    - Strips 'chr' prefix (case-insensitive, prefix-only)
    - Maps M / chrM / chrMT → 'MT'
    - Returns None for anything not in the 25 canonical chromosomes
    """
    normalized = re.sub(r"^chr", "", chrom.strip(), flags=re.IGNORECASE)

    if normalized.upper() in _MITO_ALIASES:
        normalized = "MT"

    if normalized in _VALID_CHROMS:
        return normalized

    return None





class Genome:
    """
    Loads a FASTA genome entirely into RAM for fast, multi-core-safe access.
    """

    _COMPLEMENT_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

    def __init__(self, path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._genome = self._load_fasta(path)
        self.chrom_sizes: dict[str, int] = {
            chrom: len(seq) for chrom, seq in self._genome.items()
        }
        

    @staticmethod
    def _load_fasta(path: str) -> dict[str, str]:
        """Load FASTA file using pysam into a dict of {chrom: sequence}."""
        fasta = pysam.FastaFile(path)
        genome = {}
        try:
            for chrom in fasta.references:
                norm = normalize_chrom(chrom)
                if norm is not None:
                    genome[norm] = fasta.fetch(chrom)
        finally:
            fasta.close()
        return genome

    def _reverse_complement(self, seq: str) -> str:
        """Reverse complement a DNA sequence."""
        return seq.translate(self._COMPLEMENT_TABLE)[::-1]

    def get_seq_window(
        self, 
        chrom: str,
        start: int,
        end: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fetch a genomic window [start, end) (0-based, half-open).

        Out-of-bounds edges are N-padded so the output is always exactly
        ``end - start`` tokens long.

        Returns
        -------
        (fwd_tokens, rev_tokens) : tuple of np.ndarray, shape (end-start,)
            Forward strand and its reverse complement, both tokenized.
        """
        chrom = normalize_chrom(chrom)

        if chrom not in self._genome:
            available = list(self.chrom_sizes.keys())[:10]
            raise ValueError(
                f"Chromosome '{chrom}' not found in genome. "
                f"Available (first 10): {available}"
            )

        chrom_size = self.chrom_sizes[chrom]
        raw_seq = self._genome[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            raw_seq = "N" * (-start) + raw_seq           # left pad
        if end > chrom_size:
            raw_seq = raw_seq + "N" * (end - chrom_size)  # right pad

        fwd_tokens = self.tokenizer(raw_seq)
        rev_tokens = self.tokenizer(self._reverse_complement(raw_seq))

        return (fwd_tokens, rev_tokens)

    

    def get_seq_vep(
        self,
        chr: str,
        variant_pos: int,
        ref: str,
        alt: str,
        seq_len: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Load sequences centered on a variant for impact prediction (VEP).

        The variant is placed at index ``seq_len // 2`` of the window:

            [ ← half_seq → ] [ variant ] [ ← half_seq → ]

        Only the forward strand is mutated; the reverse strand is derived by
        reverse-complementing the forward sequence, which correctly handles
        both odd and even ``seq_len`` without special-casing.

        Parameters
        ----------
        chr : str
            Chromosome name (any common format, e.g. "1", "chr1").
        variant_pos : int
            1-based position of the variant on the chromosome.
        ref : str
            Expected reference allele (validated against the FASTA).
        alt : str
            Alternate allele to substitute.
        seq_len : int
            Total output sequence length.

        Returns
        -------
        (ref_fwd, ref_rev), (alt_fwd, alt_rev)
            Each is a tokenized np.ndarray of shape (seq_len,).

        Raises
        ------
        ValueError
            If the ref allele in the FASTA does not match `ref`, or if the
            chromosome is not found.
        """
        chrom = normalize_chrom(chr)
        if chrom not in self._genome:
            available = list(self.chrom_sizes.keys())[:10]
            raise ValueError(
                f"Chromosome '{chrom}' not found in genome. "
                f"Available (first 10): {available}"
            )

        half = seq_len // 2

        # Convert variant_pos to 0-based and compute window boundaries
        var_idx = variant_pos - 1
        start = var_idx - half
        end = start + seq_len

        chrom_size = self.chrom_sizes[chrom]
        raw_seq = self._genome[chrom][max(start, 0) : min(end, chrom_size)]

        if start < 0:
            raw_seq = "N" * (-start) + raw_seq          # left pad
        if end > chrom_size:
            raw_seq = raw_seq + "N" * (end - chrom_size) # right pad

        # Index of the variant within the extracted window
        center = half

        # Validate ref allele (case-insensitive)
        actual_ref = raw_seq[center : center + len(ref)]
        if actual_ref.upper() != ref.upper():
            raise ValueError(
                f"Ref allele mismatch at {chrom}:{variant_pos}: "
                f"expected '{ref}', found '{actual_ref}' in FASTA"
            )

        # Build alt forward sequence: splice in alt at center
        alt_fwd = raw_seq[:center] + alt + raw_seq[center + len(ref):]

        # Trim or pad the alt sequence to keep output length == seq_len
        if len(alt_fwd) > seq_len:
            alt_fwd = alt_fwd[:seq_len]
        elif len(alt_fwd) < seq_len:
            alt_fwd = alt_fwd + "N" * (seq_len - len(alt_fwd))

        ref_fwd = raw_seq  # already exactly seq_len

        ref_fwd_tok = self.tokenizer(ref_fwd)
        ref_rev_tok = self.tokenizer(self._reverse_complement(ref_fwd))
        alt_fwd_tok = self.tokenizer(alt_fwd)
        alt_rev_tok = self.tokenizer(self._reverse_complement(alt_fwd))

        return (ref_fwd_tok, ref_rev_tok), (alt_fwd_tok, alt_rev_tok)

    def __repr__(self) -> str:
        total_bp = sum(self.chrom_sizes.values())
        return (
            f"Genome({len(self.chrom_sizes)} chromosomes, "
            f"{total_bp:,} bp total)"
        )


if __name__ == "__main__":
    import sys

    FA = "/Users/thanhdng/Documents/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    SEQ_LEN = 1024

    print("Loading genome...")
    tok = Tokenizer()
    g = Genome(FA, tok)
    print(g)

    # ------------------------------------------------------------------
    # Test 1: get_seq_window shape and RC correctness
    # ------------------------------------------------------------------
    print("\n[Test 1] get_seq_window")
    fwd, rev = g.get_seq_window("1", 1_000_000, 1_000_000 + SEQ_LEN)
    assert fwd.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {fwd.shape}"
    assert rev.shape == (SEQ_LEN,), f"Expected ({SEQ_LEN},), got {rev.shape}"
    rc_of_fwd = tok(g._reverse_complement(tok.decode(fwd)))
    assert (rc_of_fwd == rev).all(), "rev should be RC of fwd"
    print(f"  fwd shape: {fwd.shape}  rev shape: {rev.shape}  RC check: OK")

    # ------------------------------------------------------------------
    # Test 2: get_seq_window with N-padding (start < 0)
    # ------------------------------------------------------------------
    print("\n[Test 2] get_seq_window with left N-padding")
    fwd_pad, rev_pad = g.get_seq_window("1", -10, SEQ_LEN - 10)
    assert fwd_pad.shape == (SEQ_LEN,)
    assert (fwd_pad[:10] == tok.pad_index).all(), "first 10 tokens should be N (pad)"
    print(f"  First 10 tokens (expect all N={tok.pad_index}): {fwd_pad[:10].tolist()}  OK")

    # ------------------------------------------------------------------
    # Test 3: get_seq_vep — shape, only center differs, RC correctness
    # ------------------------------------------------------------------
    print("\n[Test 3] get_seq_vep")
    CHR, POS = "1", 230_710_048
    actual_ref = g._genome["1"][POS - 1].upper()
    alt = "T" if actual_ref != "T" else "C"
    print(f"  Using REF={actual_ref!r}  ALT={alt!r}  at {CHR}:{POS}")

    (ref_fwd, ref_rev), (alt_fwd, alt_rev) = g.get_seq_vep(CHR, POS, actual_ref, alt, SEQ_LEN)
    assert ref_fwd.shape == (SEQ_LEN,)
    assert ref_rev.shape == (SEQ_LEN,)
    assert alt_fwd.shape == (SEQ_LEN,)
    assert alt_rev.shape == (SEQ_LEN,)

    center = SEQ_LEN // 2
    assert ref_fwd[center] != alt_fwd[center], "center base must differ"
    assert (ref_fwd[:center] == alt_fwd[:center]).all(), "left flank must be identical"
    assert (ref_fwd[center + 1:] == alt_fwd[center + 1:]).all(), "right flank must be identical"

    rc_ref = tok(g._reverse_complement(tok.decode(ref_fwd)))
    assert (rc_ref == ref_rev).all(), "ref_rev should be RC of ref_fwd"
    rc_alt = tok(g._reverse_complement(tok.decode(alt_fwd)))
    assert (rc_alt == alt_rev).all(), "alt_rev should be RC of alt_fwd"
    print(f"  Shape OK, center differs, flanks identical, RC check: OK")

    # ------------------------------------------------------------------
    # Test 4: ref mismatch raises ValueError
    # ------------------------------------------------------------------
    print("\n[Test 4] ref mismatch raises ValueError")
    wrong_ref = "T" if actual_ref != "T" else "C"
    try:
        g.get_seq_vep(CHR, POS, wrong_ref, alt, SEQ_LEN)
        print("  FAIL: no exception raised")
        sys.exit(1)
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")

    print("\nAll tests passed.")

    # ------------------------------------------------------------------
    # Speed test: 100k get_seq_window calls
    # ------------------------------------------------------------------
    import time, random

    N = 100_000
    chrom_size = g.chrom_sizes["1"]
    rng = random.Random(42)
    starts = [rng.randint(0, chrom_size - SEQ_LEN) for _ in range(N)]

    print(f"\n[Speed] get_seq_window x{N:,} (seq_len={SEQ_LEN})")
    t0 = time.perf_counter()
    for s in starts:
        g.get_seq_window("1", s, s + SEQ_LEN)
    dt = time.perf_counter() - t0
    print(f"  Total: {dt:.2f}s  |  {N/dt:,.0f} calls/s  |  {dt/N*1e6:.2f} µs/call")

    # ------------------------------------------------------------------
    # Speed test: 100k get_seq_vep calls (SNP: single-base substitution)
    # ------------------------------------------------------------------
    positions = [rng.randint(SEQ_LEN // 2 + 1, chrom_size - SEQ_LEN // 2) for _ in range(N)]
    bases = list("ATGC")

    print(f"\n[Speed] get_seq_vep    x{N:,} (seq_len={SEQ_LEN})")
    t0 = time.perf_counter()
    for pos in positions:
        ref_base = g._genome["1"][pos - 1].upper()
        if ref_base not in bases:
            continue  # skip ambiguous/N positions
        alt_base = next(b for b in bases if b != ref_base)
        g.get_seq_vep("1", pos, ref_base, alt_base, SEQ_LEN)
    dt = time.perf_counter() - t0
    print(f"  Total: {dt:.2f}s  |  {N/dt:,.0f} calls/s  |  {dt/N*1e6:.2f} µs/call")