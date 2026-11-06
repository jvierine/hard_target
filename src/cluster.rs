//! MPI distribution of independent matched-filter delay rows.
//!
//! All ranks call collectively with identical input arrays/settings. MPI is
//! called only by the invoking thread; Rayon/CUDA workers never call MPI.
//! Results stay rank-local: use unique rank files, or gather explicitly. CFAR
//! must run on the complete range context, not independently on these shards.
use crate::{search_cpu, validate, Output, Settings};
use anyhow::{ensure, Result};
use mpi::{collective::SystemOperation, traits::*};
use num_complex::Complex32;
use std::ops::Range;

#[derive(Clone, Copy)]
pub enum Backend {
    Cpu,
    /// Ordinal within the scheduler-provided CUDA_VISIBLE_DEVICES namespace.
    #[cfg(feature = "cuda")]
    Cuda {
        device: usize,
        memory_budget_bytes: usize,
        tile_ranges: usize,
    },
}

pub struct Shard {
    /// Indices into the original settings.delays array, preserving its order.
    pub rows: Range<usize>,
    pub output: Option<Output>,
}

pub fn partition(rows: usize, rank: usize, ranks: usize) -> Result<Range<usize>> {
    ensure!(ranks > 0 && rank < ranks, "invalid MPI rank/size");
    let base = rows / ranks;
    let extra = rows % ranks;
    let start = rank * base + rank.min(extra);
    Ok(start..start + base + usize::from(rank < extra))
}

/// Collective error agreement prevents one ordinary backend error leaving peers
/// blocked at a later collective. Process death/panics require launcher teardown.
fn agree<C: CommunicatorCollectives>(world: &C, ok: bool) -> Result<()> {
    let mut all_ok = 0_i32;
    world.all_reduce_into(&i32::from(ok), &mut all_ok, SystemOperation::min());
    ensure!(
        all_ok == 1,
        "at least one MPI rank failed; inspect rank-local diagnostics"
    );
    Ok(())
}

/// `threads` is a per-rank allocation, not the node CPU count. Call after MPI
/// initialization with at least Threading::Funneled support, from its main thread.
/// The caller loads/broadcasts identical IQ/settings before invoking this API.
/// No environment, global thread pool, GPU visibility, or output path is changed.
pub fn search<C: Communicator + CommunicatorCollectives>(
    world: &C,
    tx: &[Complex32],
    rx: &[Complex32],
    settings: &Settings,
    threads: usize,
    backend: Backend,
) -> Result<Shard> {
    let valid = validate(tx, rx, settings).and_then(|_| {
        ensure!(threads > 0, "per-rank CPU allocation must be positive");
        Ok(())
    });
    let agreement = agree(world, valid.is_ok());
    valid?;
    agreement?;
    // Reject inconsistent grids before assigning any work. Full input equality
    // remains a caller contract, avoiding a second O(input size) scan/hash here.
    let shape = [
        settings.delays.len() as u64,
        settings.display_bins as u64,
        settings.doppler_rates_hz_s.len() as u64,
        tx.len() as u64,
        rx.len() as u64,
    ];
    let mut minimum = [0_u64; 5];
    let mut maximum = [0_u64; 5];
    world.all_reduce_into(&shape, &mut minimum, SystemOperation::min());
    world.all_reduce_into(&shape, &mut maximum, SystemOperation::max());
    ensure!(
        minimum == maximum,
        "MPI ranks supplied inconsistent input shapes"
    );
    let rows = partition(
        settings.delays.len(),
        world.rank() as usize,
        world.size() as usize,
    )?;
    let result = (|| -> Result<Option<Output>> {
        if rows.is_empty() {
            return Ok(None);
        }
        let mut local = settings.clone();
        local.delays = settings.delays[rows.clone()].to_vec();
        match backend {
            Backend::Cpu => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()?;
                Ok(Some(pool.install(|| search_cpu(tx, rx, &local))?))
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda {
                device,
                memory_budget_bytes,
                tile_ranges,
            } => {
                let mut bank = crate::gpu::GpuBank::new(device)?;
                bank.memory_budget_bytes = memory_budget_bytes;
                bank.tile_ranges = tile_ranges;
                Ok(Some(bank.search(tx, rx, &local)?))
            }
        }
    })();
    let agreement = agree(world, result.is_ok());
    let output = result?;
    agreement?;
    Ok(Shard { rows, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_row_owned_exactly_once_in_order() {
        for n in 0..40 {
            for ranks in 1..16 {
                let rows: Vec<_> = (0..ranks)
                    .flat_map(|r| partition(n, r, ranks).unwrap())
                    .collect();
                assert_eq!(rows, (0..n).collect::<Vec<_>>());
            }
        }
        assert!(partition(5, 0, 0).is_err());
        assert!(partition(5, 3, 3).is_err());
    }
}
