# Container definitions

Both images used in this work are built from these two Apptainer definition
files. They are copied here verbatim so the environment can be rebuilt from
this repository alone, and they are kept in step with the build repository at
<https://github.com/farmer-vn/olivia_containers>, commit `419a61319291`, which
also holds the submission wrappers and the smoke tests.

| file | image | architecture | used for |
|---|---|---|---|
| `seq2fun.def` | `seq2fun.sif` | arm64 | training and inference on gh200 GPU nodes |
| `seq2fun_cpu.def` | `seq2fun_cpu.sif` | amd64 | every analysis and figure script |

Architecture follows the machine that runs the build, so the two images must be
built on different nodes. Building the CPU image on a GPU node yields a second
arm64 image that will not execute on a CPU partition.

```bash
apptainer build seq2fun.sif      containers/seq2fun.def        # on a gh200 node
apptainer build seq2fun_cpu.sif  containers/seq2fun_cpu.def    # on an x86 node
```

Each definition carries a `%test` section that Apptainer runs as part of the
build, so a build that reports success has already checked the architecture,
the CUDA state and the whole import stack. The CPU image additionally checks
that an `N` base encodes as an all-zero one-hot column rather than sharing a
channel with `A`.

Running something inside an image.

```bash
apptainer exec --bind /cluster:/cluster seq2fun_cpu.sif python <script>.py
apptainer exec --nv --pwd /workspace --bind "$PWD":/workspace --bind /cluster:/cluster \
    seq2fun.sif python -m src.seq2fun.trainer ...
```
