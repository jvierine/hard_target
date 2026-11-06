# Cluster execution

The library supports optional MPI (`--features mpi`) with either the SIMD/Rayon
CPU backend or CUDA (`--features mpi,cuda`). `cluster::search` distributes
contiguous delay rows over the supplied MPI communicator. Each rank retains its
output and original row indices; empty ranks are valid. Every rate/frequency is
searched for each assigned delay. Concatenate shards in rank order to recover
the serial bank. No shared output file is opened by the library.

This API distributes the matched-filter bank, not the campaign dispatcher or
range CFAR. CFAR requires training cells across rank boundaries: gather the bank
or distribute complete independent recordings before performing detection.
Do not run range CFAR independently on truncated delay shards.

## Resource and execution contract

* Initialize MPI with `Threading::Funneled` (or stronger), call collectives on
  the initializing thread, and keep the MPI universe alive until processing ends.
* All ranks must call `cluster::search` with identical IQ and settings. The caller
  loads or broadcasts input. Shapes are checked collectively; content identity
  is a caller responsibility. Ordinary validation/backend failures are reported
  collectively. Panics/process death require launcher abort; this is not ULFM.
* Pass the scheduler's **per-task** CPU allocation as `threads`. Each CPU rank
  uses its own bounded Rayon pool. Do not infer the full node core count. Bind
  ranks/threads with the scheduler; use `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` for other linked libraries.
* CUDA device indices refer to CUDA's visible namespace. Allocate/bind GPUs in
  the scheduler and pass an explicit ordinal to `Backend::Cuda`. With one GPU
  exposed per rank use 0; with multiple exposed devices assign unique ordinals
  per node. The library does not override CUDA_VISIBLE_DEVICES, select a hostname,
  oversubscribe GPUs, or silently switch backends. Set an explicit memory budget
  and tile size. The budget bounds library allocations, not all driver overhead.
* Use rank-specific output files and a separate merge step. Serial HDF5 is used;
  concurrent writes to one HDF5 file are unsupported. Publish a completion
  manifest only after every rank succeeds. Use job scratch for intermediates.
* Build on a compatible compute environment with HDF5 and the same MPI ABI used
  at runtime. CUDA needs compatible NVIDIA driver/cuFFT libraries. Build without
  `-C target-cpu=native` when compute nodes differ. SIMD uses runtime CPU dispatch.
  CPU builds require no CUDA runtime; non-MPI builds require no MPI runtime.
* This does not install software on nodes, SSH to workers, start background
  services, or submit jobs. The existing Arecibo workstation dispatcher is a
  separate campaign tool and is not the MPI/Slurm execution path.

## Build and validate

```sh
cargo test -p hard-target --release --features mpi
cargo build -p hard-target --release --features mpi --example validate_mpi
RAYON_NUM_THREADS=1 mpirun -n 2 target/release/examples/validate_mpi
RAYON_NUM_THREADS=1 mpirun -n 4 target/release/examples/validate_mpi
cargo build -p hard-target --release --features mpi,cuda --example validate_mpi
```

The MPI test compares distributed signed-IQ search results against serial CPU
results for uneven partitions and more ranks than rows, and injects a rank-local
validation failure to check collective error propagation. `validate_mpi` can
also exercise CUDA when `RUISDAP_TEST_CUDA=1` and `RUISDAP_CUDA_DEVICE` explicitly
selects a scheduler-bound device. Do not run multiple ranks on one GPU unless
intentionally testing oversubscription. Single-node CPU MPI testing is not a
multi-node interconnect or scheduler certification.

Slurm example (adjust partition/account/modules for the target cluster; no
site-specific account is embedded):

```sh
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=4G
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Build before allocation. Use the site's MPI-compatible srun plugin.
srun --cpu-bind=cores ./target/release/examples/validate_mpi
```

For GPU jobs request one GPU per task and bind it using the site's scheduler
configuration, then pass `Backend::Cuda { device: 0, ... }` when each task sees
exactly its allocated GPU. Do not manually rewrite the scheduler's visibility.
