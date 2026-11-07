"""Is the held-out chromosome really held out, and did donors land on it?

Three things the flank ablation depends on and none of which it checks:

  1. test_chromosome() reads one arrow shard. That is correct only while the
     split is unsharded, so confirm it per species rather than assume it.
  2. The chromosome it returns must be absent from the training split --
     otherwise random_source donors come from sequence the source model
     memorised, and rescue measures recall rather than transferable grammar.
  3. The donors actually drawn must lie on that chromosome.
"""
import glob
import os
import sys

import pandas as pd
import pyarrow.ipc as ipc

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# ablation_dataset imports torch, which CPU nodes do not carry. Lift the real
# function's source out of the file and exec it, so this checks the shipped
# implementation rather than a paraphrase of it.
import ast
_src = open(os.path.join(ROOT, "analysis", "flank_ablation", "src",
                         "ablation_dataset.py")).read()
_fn = next(n for n in ast.parse(_src).body
           if isinstance(n, ast.FunctionDef) and n.name == "test_chromosome")
_ns = {"os": os}
exec(compile(ast.Module([_fn], []), "<test_chromosome>", "exec"), _ns)
test_chromosome = _ns["test_chromosome"]

D = "/cluster/projects/nn9780k/dat/training_data/seq2fun/binary"
DATA = {"human": "hf_ENTEX_423_beds", "pig": "hf_pig_beds",
        "cattle": "hf_cattle_beds", "chicken": "hf_chicken_beds",
        "salmon": "hf_salmon_beds", "zebrafish": "hf_zebrafish_beds"}
CACHE = os.path.join(ROOT, "analysis", "flank_ablation", "data", "cache")


def chroms(data_dir, split):
    """Every chromosome in a split, reading ALL shards."""
    out = set()
    for f in sorted(glob.glob(os.path.join(data_dir, split, "*.arrow"))):
        with open(f, "rb") as fh:
            out |= set(ipc.open_stream(fh).read_all()["chrom"].to_pylist())
    return out


def main():
    rows = []
    for sp, d in DATA.items():
        dd = os.path.join(D, d)
        shards = len(glob.glob(os.path.join(dd, "test", "*.arrow")))
        reported = test_chromosome(dd)                 # what the code uses
        true_test = chroms(dd, "test")                 # all shards
        train = chroms(dd, "train")
        val = chroms(dd, "validation")
        rows.append({
            "species": sp, "test_shards": shards,
            "reported": reported,
            "test_is_single": len(true_test) == 1,
            "reported_matches": reported in true_test and len(true_test) == 1,
            "leaks_into_train": reported in train,
            "leaks_into_val": reported in val,
            "n_train_chroms": len(train),
            "validation_chrom": test_chromosome(dd, "validation"),
        })
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("=== held-out chromosome per species ===")
    print(t.to_string(index=False))

    print("\n=== did random_source donors land on the source's test chromosome? ===")
    bad = 0
    for f in sorted(glob.glob(os.path.join(CACHE, "*_to_*",
                                           "donors__random_source_*.tsv.gz"))):
        src = os.path.basename(os.path.dirname(f)).split("_to_")[0]
        want = test_chromosome(os.path.join(D, DATA[src]))
        d = pd.read_csv(f, sep="\t", dtype={"donor_chrom": str})
        used = set(d.donor_chrom) - {"-"}     # "-" means no donor was found
        ok = used == {str(want)}
        if not ok:
            bad += 1
            print(f"  MISMATCH {os.path.basename(os.path.dirname(f))}: "
                  f"want {want}, got {sorted(used)[:5]}")
    print(f"  {'all donors on the held-out chromosome' if not bad else f'{bad} mismatched'}")

    print("\n=== did random_same donors avoid the target's test chromosome? ===")
    bad = 0
    for f in sorted(glob.glob(os.path.join(CACHE, "*_to_*",
                                           "donors__random_same_k*.tsv.gz"))):
        direction = os.path.basename(os.path.dirname(f))
        src, tgt = direction.split("_to_")
        if src == tgt:
            continue
        barred = str(test_chromosome(os.path.join(D, DATA[tgt])))
        d = pd.read_csv(f, sep="\t", dtype={"donor_chrom": str})
        hit = (d.donor_chrom == barred).sum()
        if hit:
            bad += 1
            print(f"  LEAK {direction}: {hit} donors on barred {barred}")
    print(f"  {'no donor on a barred chromosome' if not bad else f'{bad} directions leak'}")


if __name__ == "__main__":
    main()
