"""Regenerate every manuscript artifact under the final convention, and diff.

Final convention = N-flank exclusion (interval level) + 50-positive floor
(target-track level, counted on the kept intervals), applied identically to
within-species and cross-species evaluation.
"""
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MA = os.path.join(ROOT, "manuscript", "analysis")
OUT = os.path.join(MA, "out")
FINAL_TS = os.path.join(ROOT, "eval_results", "test_set_final")
sys.path.insert(0, MA)


def run(script, env=None, label="", stdout_to=None):
    e = dict(os.environ, REPO_ROOT=ROOT, **(env or {}))
    r = subprocess.run([sys.executable, script], cwd=MA, env=e,
                       capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"  {script:34s} {status} {label}", flush=True)
    if r.returncode:
        print("    " + "\n    ".join((r.stderr or "").strip().split("\n")[-5:]))
    elif stdout_to:
        # stats_gradient.py reports on stdout and writes no file of its own, so
        # without this the committed report keeps whatever convention it was
        # last run under while the margins beside it move on.
        with open(stdout_to, "w") as fh:
            fh.write(r.stdout)
        print(f"  {'':34s}    -> {os.path.relpath(stdout_to, ROOT)}")
    return r.returncode == 0


def main():
    print("== 1. per-track tables under the final convention")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "analysis", "nflank", "src",
                                 "build_final_test_set.py")],
                   check=True, env=dict(os.environ, REPO_ROOT=ROOT))

    print("\n== 2. aggregate cross-species margins")
    for cs, name in ((os.path.join(ROOT, "eval_results", "cross_species"),
                      "cross_species_margins_nofloor.csv"),
                     (os.path.join(ROOT, "eval_results", "cross_species_final"),
                      "cross_species_margins_final.csv")):
        subprocess.run([sys.executable, "aggregate_cross_species.py"], cwd=MA,
                       check=True,
                       env=dict(os.environ, REPO_ROOT=ROOT, CS_DIR=cs,
                                MARGINS_NAME=name))
    shutil.copyfile(os.path.join(OUT, "cross_species_margins_final.csv"),
                    os.path.join(OUT, "cross_species_margins.csv"))
    print("  final margins are now primary")

    print("\n== 3. regenerate artifacts")
    # Both variables for every script. 3.cross_species.py reads the per-match
    # tree directly for panels c and d, and s2/s4 import its loaders, so a
    # TEST_SET_DIR-only environment leaves those panels on the old convention.
    env = {"TEST_SET_DIR": FINAL_TS,
           "CS_DIR": os.path.join(ROOT, "eval_results", "cross_species_final")}
    for s in ["ts1.within_species.py", "2.model_performance.py",
              "3.cross_species.py", "t1.margin_summary.py",
              "ts2.chicken_source.py", "ts3.margin_sensitivity.py",
              "s4.margin_baseline.py", "s2.absolute_transfer.py",
              "s3.transfer_retention.py"]:
        run(s, env)
    run("stats_gradient.py", env,
        stdout_to=os.path.join(OUT, "stats_report.txt"))


if __name__ == "__main__":
    main()
