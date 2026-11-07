"""Test-set dataset with the flank replaced, central k bp kept real.

Mirrors Seq2FunDataset's window arithmetic exactly -- center = (start+end)//2,
window = [center - L//2, center + L//2) -- so the `intact` condition is
token-identical to the main pipeline rather than merely similar. assert_intact()
checks that against the real Seq2FunDataset at run time.

Conditions
  intact         the window untouched; the regression reference
  shuffled       each flank dinucleotide-shuffled in place, so length, GC and
                 dinucleotide composition are preserved and only the arrangement
                 is destroyed
  random_same    flanks taken from a random locus in the TARGET genome,
                 GC-matched to the flank they replace
  random_source  flanks taken from a random locus in the SOURCE genome, around
                 the target's real central k bp

N is never used as filler: 19 kb of N is far outside anything the model saw in
training, which is what made an earlier N-padding attempt uninterpretable.
Donor loci are rejected if they carry appreciable N for the same reason.

The reverse strand is recomputed from the modified forward sequence. Reusing
the genome's original reverse strand would silently score a different sequence
on one of the two strands the pipeline averages.
"""
import os
import sys

import numpy as np
import torch

# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from src.bio.genome import Genome, Tokenizer          # noqa: E402
from src.data.seq2fun import Seq2FunDataset          # noqa: E402
from torch.utils.data import Dataset                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shuffle import dinuc_shuffle, gc_fraction        # noqa: E402

CONDITIONS = ("intact", "shuffled", "random_same", "random_source")

# Donor acceptance. Tolerance escalates rather than dropping an interval: a
# donor within 2% GC is taken if found, otherwise the search widens to 3%, and
# only if that also fails is the closest candidate seen kept. Every interval
# therefore gets a real flank, and the tier it was satisfied at is recorded so
# the match rate at each tolerance is reportable per direction.
GC_TIERS = (0.02, 0.03)
TRIES_PER_TIER = 40
MAX_N_FRAC = 0.01      # reject donors more than 1% N overall
MAX_N_RUN = 50         # reject donors with an N run this long (assembly gaps)


class FlankAblationDataset(Dataset):
    def __init__(self, data_dir, genome_path, sequence_length, condition,
                 keep_bp=1000, seed=0, donor_genome_path=None, split="test",
                 donor_exclude_chrom=None, donor_only_chrom=None, log_dir=None):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        if condition == "random_source" and donor_genome_path is None:
            raise ValueError("random_source needs --donor-genome (the source species)")
        if keep_bp > sequence_length:
            raise ValueError("keep_bp cannot exceed the model's context length")

        from datasets import load_from_disk
        self.rows = load_from_disk(data_dir)[split]
        self.L = sequence_length
        self.half = sequence_length // 2
        self.keep = keep_bp
        self.condition = condition
        self.seed = seed
        self.tok = Tokenizer()
        self.genome_path = genome_path
        self.donor_path = donor_genome_path if condition == "random_source" else (
            genome_path if condition == "random_same" else None)
        self.genome = None
        self.donor = None
        # Chromosome barred as a donor. The test split tiles one whole
        # chromosome, so every donor drawn from it would overlap some evaluated
        # interval's own 20 kb window; barring the chromosome is the same rule,
        # stated once. For random_source the donor genome is a different
        # species, so this bars that species' held-out chromosome instead,
        # keeping donors inside sequence the source model trained on.
        self.exclude_chrom = donor_exclude_chrom
        # random_source restricts donors to the SOURCE species' held-out
        # chromosome. Drawing them from its training chromosomes would make the
        # source flank sequence the model has seen, while the random_same flank
        # comes from a genome it has never seen at all -- so the contrast would
        # carry a seen-vs-unseen difference on top of the genome-of-origin one
        # it exists to isolate. Overlap with the source's own test windows is
        # irrelevant: the evaluated intervals are the target's.
        self.only_chrom = donor_only_chrom
        # Donor provenance is written straight to disk, one file per dataloader
        # worker, rather than accumulated on the instance: with num_workers > 0
        # each worker mutates its own copy of this object and anything appended
        # to a list here is discarded when the worker exits.
        self.log_dir = log_dir
        self._log = None

    # genomes are opened lazily so DataLoader workers each hold their own
    def _open_log(self):
        if (self.log_dir is None or self._log is not None
                or self.condition == "intact"):
            return                      # intact replaces nothing; no donors to log
        import atexit
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        wid = 0 if info is None else info.id
        os.makedirs(self.log_dir, exist_ok=True)
        self._log = open(os.path.join(self.log_dir, f"donors_w{wid}.tsv"), "w")
        self._log.write("row\tside\tdonor_chrom\tdonor_start\tgc_orig\tgc_new\tgc_tier\n")
        # A dataloader worker is torn down without anyone closing this handle,
        # so whatever is still buffered is lost -- which silently dropped 3.3%
        # of records, and left intact runs with a header-only, zero-byte file.
        # atexit runs on normal worker shutdown; the periodic flush bounds the
        # loss if a worker is killed instead.
        atexit.register(self._close_log)

    def _close_log(self):
        if self._log is not None and not self._log.closed:
            self._log.flush()
            self._log.close()

    def _ensure(self):
        self._open_log()
        if self.genome is None:
            self.genome = Genome(self.genome_path, self.tok)
        if self.donor_path is not None and self.donor is None:
            self.donor = (self.genome if self.donor_path == self.genome_path
                          else Genome(self.donor_path, self.tok))

    def _n_run_ok(self, tokens):
        """No N run of MAX_N_RUN or more -- i.e. no assembly gap in the donor."""
        if not (tokens == 0).any():
            return True
        nz = np.flatnonzero(tokens != 0)
        edges = np.concatenate(([-1], nz, [len(tokens)]))
        return int(np.diff(edges).max()) - 1 < MAX_N_RUN

    def _donor_flank(self, n_bp, want_gc, rng):
        """A real, GC-matched, gap-free donor stretch of `n_bp`.

        Returns (tokens, chrom, start, tier) where tier is the GC tolerance it
        satisfied, or inf when no tier was met and the closest was kept.
        """
        sizes = self.donor.chrom_sizes
        if self.only_chrom is not None:
            names = [self.only_chrom] if sizes.get(self.only_chrom, 0) > n_bp + 1000 else []
        else:
            names = [c for c, L in sizes.items()
                     if L > n_bp + 1000 and c != self.exclude_chrom]
        if not names:
            return None
        # Draw the chromosome in proportion to its number of valid start
        # offsets, so chromosome and position together are a uniform draw over
        # every legal locus. Choosing uniformly among chromosome names instead
        # gives a short chromosome the same expected donor count as a long one,
        # which over-samples it per base -- 170-fold for chicken, whose
        # microchromosomes are gene-dense and unrepresentative of the genome.
        w = None
        if len(names) > 1:
            u = np.array([sizes[c] - n_bp for c in names], dtype=float)
            w = u / u.sum()
        best = None
        best_d = np.inf
        for tier in GC_TIERS:
            for _ in range(TRIES_PER_TIER):
                c = (names[rng.integers(len(names))] if w is None
                     else str(rng.choice(names, p=w)))
                st = int(rng.integers(0, sizes[c] - n_bp))
                tok = self.tok(self.donor._genome[c][st:st + n_bp])
                if float((tok == 0).mean()) > MAX_N_FRAC or not self._n_run_ok(tok):
                    continue
                d = abs(gc_fraction(tok) - want_gc)
                if d < best_d:
                    best, best_d = (tok, c, st), d
                if d <= tier:
                    return (*best, tier)
        return (*best, np.inf) if best is not None else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        self._ensure()
        r = self.rows[i]
        center = (r["start"] + r["end"]) // 2
        fwd, _ = self.genome.get_seq_window(r["chrom"], center - self.half,
                                            center + self.half)
        fwd = fwd.copy()

        if self.condition != "intact":
            rng = np.random.default_rng((self.seed, i))
            lo = (self.L - self.keep) // 2
            hi = lo + self.keep
            for a, b in ((0, lo), (hi, self.L)):
                if b <= a:
                    continue
                orig = fwd[a:b]
                side = "L" if a == 0 else "R"
                c, st = "-", -1
                if self.condition == "shuffled":
                    new, tier = dinuc_shuffle(orig, rng), 0.0
                else:
                    # random_source matches the TARGET flank's GC using a donor
                    # from the SOURCE genome, so the only thing that changes
                    # between random_same and random_source is which genome the
                    # real sequence came from.
                    got = self._donor_flank(b - a, gc_fraction(orig), rng)
                    if got is None:
                        new, tier = dinuc_shuffle(orig, rng), np.nan
                        c, st = "-", -1
                    else:
                        new, c, st, tier = got
                if self._log is not None:
                    self._n_logged = getattr(self, "_n_logged", 0) + 1
                    if self._n_logged % 20000 == 0:
                        self._log.flush()
                    self._log.write(
                        f"{i}\t{side}\t{c if self.condition != 'shuffled' else '-'}"
                        f"\t{st if self.condition != 'shuffled' else -1}"
                        f"\t{gc_fraction(orig):.6f}\t{gc_fraction(new):.6f}\t{tier}\n")
                fwd[a:b] = new

        rev = self.tok(self.genome._reverse_complement(self.tok.decode(fwd)))
        return {"fwd_ids": torch.from_numpy(fwd).long(),
                "rev_ids": torch.from_numpy(rev).long(),
                "labels": torch.tensor(r["labels"], dtype=torch.float32)}


def assert_intact(data_dir, genome_path, sequence_length, n=64):
    """intact must be token-identical to the pipeline's own dataset."""
    ref = Seq2FunDataset(split="test", data_dir=data_dir, genome_path=genome_path,
                         sequence_length=sequence_length)
    new = FlankAblationDataset(data_dir, genome_path, sequence_length, "intact")
    idx = np.linspace(0, len(ref) - 1, n).astype(int)
    for i in idx:
        a, b = ref[int(i)], new[int(i)]
        for k in ("fwd_ids", "rev_ids", "labels"):
            if not torch.equal(a[k], b[k]):
                raise SystemExit(f"intact differs from Seq2FunDataset at row {i} ({k})")
    return len(idx)


def test_chromosome(data_dir, split="test"):
    """The single chromosome the split occupies, or None if it spans several.

    Every species' test split here is one held-out chromosome, which is why a
    donor drawn from it would overlap some evaluated interval's own window.
    """
    import glob

    import pyarrow.ipc as ipc
    # Read every shard. The test and validation splits are single-shard today,
    # but train is not, and a single-shard read of a sharded split would report
    # one chromosome from the first shard and silently look like a clean
    # hold-out.
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.arrow")))
    if not files:
        raise FileNotFoundError(f"no arrow shards in {data_dir}/{split}")
    u = set()
    for path in files:
        with open(path, "rb") as fh:
            u |= set(ipc.open_stream(fh).read_all()["chrom"].to_pylist())
    return u.pop() if len(u) == 1 else None
