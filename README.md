# Hard target

Rust is the main codebase: radar range–Doppler–Doppler-rate matched filtering
with SIMD CPU kernels, Rayon, optional CUDA/cuFFT, and MPI parallelization.

```sh
cargo test --workspace --release
cargo build --release --features cuda
cargo build --release --features mpi --example validate_mpi
RAYON_NUM_THREADS=1 mpirun -n 4 target/release/examples/validate_mpi
```

MPI distributes delay rows with bounded per-rank resources. See
[CLUSTER.md](CLUSTER.md) for Slurm, GPU assignment and output handling.
CUDA and MPI require compatible system libraries; the default CPU build needs neither.

Use `search_cpu`, `gpu::GpuBank`, or `cluster::search` from the `hard_target`
crate. Input is complex IQ and a measured transmit waveform. Positive Doppler
means positive recorded-IQ frequency; delays are two-way propagation samples.
Range migration is not corrected by the fixed-delay search.

The optional `arecibo` feature adds campaign-specific HDF5 adapters using pinned
RUISDAP dependencies. It is not needed for the core library.

[`python/`](python/) contains the old manually coded reference implementation,
including its C/CUDA extensions. To install it: `cd python && pip install .`.
The Rust API is not a drop-in replacement for the old Python command line.
