# Shared infrastructure

```
snapshot.sh        freeze the code a batch will run into analysis/_runs/<name>/
run_cpu.sh         submit any analysis script to a CPU compute node from a snapshot
cpu_job.slurm      the generic CPU batch script run_cpu.sh submits
n_content.py       per-interval flank N content, per species
excluded_profile.py  what the N-excluded intervals contain (central N, labels)
```

Two rules these exist to enforce, both learned the hard way:

**Never run analysis on the login node.** CPU nodes carry the same stack
(x86_64, `/usr/bin/python3.12`, numpy, pandas, pyarrow, matplotlib) plus scipy,
which the login node lacks, so no container is needed.

**Never let a queued job read a working-tree script.** A job starts long after
submission; an edit in between silently changes what runs. `snapshot.sh` copies
the code and the submitters point jobs at that copy, with `REPO_ROOT` left
pointing at the real repository so checkpoints and datasets still resolve.
Commit before submitting so the recorded hash is clean rather than `-dirty`.

Anything deriving a path from `__file__` must switch to `REPO_ROOT` under a
snapshot, or its output lands inside the frozen code tree.
