"""Assert every donor actually came from the pool its condition requires.

This exists because the pool rule was once written into the dataset while the
job script kept passing the old argument, so the code read as correct and
sampled from the wrong chromosome. A rule that is only expressed where the
sampling happens cannot catch that; this re-derives the expectation from the
condition name and checks it against the donor coordinates that were actually
written.

The rules, and why each is what it is:

  shuffled                    no donor at all -- the flank is its own shuffle
  random_same, transfer       any target chromosome EXCEPT its test chromosome,
                              every interval of which lies inside some evaluated
                              window. The source model has never seen the target
                              genome, so any other chromosome is unseen.
  random_same, within-species the target's VALIDATION chromosome. Its training
                              chromosomes are sequence the model was fitted on.
  random_source               the source's held-out chromosome, for the same
                              reason.

Used two ways: run_ablation.py calls check_file() on the log it just wrote, so a
bad job fails at once, and `python src/verify_donor_pools.py` sweeps every log
already on disk.
"""
import glob
import os
import re
import sys

import pandas as pd

# Self-contained so it does not depend on HERE, which is defined below
# and may itself point at a code snapshot.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Outputs belong in the real analysis directory. Deriving this from
# __file__ puts them inside the code snapshot when one is in use.
HERE = os.path.join(ROOT, "analysis", "flank_ablation") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
sys.path.insert(0, os.path.join(ROOT, "analysis", "tss_strata", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NO_DONOR = "-"


def _test_chrom(data_dir, split="test"):
    import pyarrow.ipc as ipc
    with open(os.path.join(data_dir, split, "data-00000-of-00001.arrow"), "rb") as fh:
        t = ipc.open_stream(fh).read_all()
    u = set(t["chrom"].to_pylist())
    return u.pop() if len(u) == 1 else None


def expected_pool(source, target, condition):
    """(mode, chrom) where mode is 'none', 'only' or 'exclude'."""
    from manifest import DATA
    if condition == "shuffled":
        return ("none", None)
    if condition == "random_source":
        return ("only", _test_chrom(DATA[source]))
    if condition == "random_same":
        if source == target:
            v = _test_chrom(DATA[target], "validation")
            return ("only", v) if v else ("exclude", _test_chrom(DATA[target]))
        return ("exclude", _test_chrom(DATA[target]))
    raise ValueError(f"unknown condition {condition!r}")


def check_file(path, source, target, condition, raise_on_fail=True):
    """Verify one donors__*.tsv.gz. Returns (n_rows, n_violations, note)."""
    df = pd.read_csv(path, sep="\t", dtype={"donor_chrom": str})
    mode, chrom = expected_pool(source, target, condition)
    seen = df["donor_chrom"].astype(str)
    fallback = int((seen == NO_DONOR).sum())
    real = seen[seen != NO_DONOR]

    if mode == "none":
        bad = real
        rule = "no donor expected"
    elif mode == "only":
        bad = real[real != str(chrom)]
        rule = f"donors must be on chr{chrom}"
    else:
        bad = real[real == str(chrom)]
        rule = f"donors must avoid chr{chrom}"

    note = (f"{os.path.basename(path)}: {len(df)} flanks, {rule}, "
            f"{len(bad)} violation(s), {fallback} fell back to shuffle")
    if len(bad) and raise_on_fail:
        offenders = bad.value_counts().head(5).to_dict()
        raise SystemExit(
            f"DONOR POOL VIOLATION\n  {path}\n  {source}->{target} {condition}\n"
            f"  rule: {rule}\n  {len(bad)} of {len(real)} donors break it; "
            f"offending chromosomes: {offenders}")
    return len(df), len(bad), note


def main():
    pats = sorted(glob.glob(os.path.join(HERE, "data", "cache", "*", "donors__*.tsv.gz")))
    if not pats:
        print("no donor logs on disk yet")
        return 0
    bad_total = 0
    for p in pats:
        direction = os.path.basename(os.path.dirname(p))
        source, target = direction.split("_to_")
        m = re.match(r"donors__(.+)_k\d+_s\d+\.tsv\.gz$", os.path.basename(p))
        if not m:
            print(f"unparseable name, skipped: {p}")
            continue
        n, bad, note = check_file(p, source, target, m.group(1), raise_on_fail=False)
        bad_total += bad
        print(("FAIL " if bad else "ok   ") + f"{direction}  {note}")
    print(f"\n{len(pats)} log(s) checked, {bad_total} violation(s)")
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())
