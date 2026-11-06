//! Small numerical/collective smoke test; no archive I/O or campaign processing.
use anyhow::{ensure, Result};
use mpi::{environment::Threading, traits::*};
use num_complex::Complex32 as C;
use hard_target::{
    cluster::{self, Backend},
    search_cpu, Settings,
};
fn run(world: &impl CommunicatorCollectives) -> Result<()> {
    let tx: Vec<_> = (0..256)
        .map(|i| C::from_polar(1., ((i * i * 17 + i * 31) % 997) as f32))
        .collect();
    let mut rx = vec![C::default(); 280];
    for (i, z) in tx.iter().enumerate() {
        rx[i + 3] = *z * C::from_polar(1., -2. * std::f32::consts::PI * 17. * i as f32 / 256.);
    }
    for count in [1, 7, 23] {
        let s = Settings {
            sample_rate_hz: 256.,
            decimation: 1,
            delays: (0..count).collect(),
            doppler_rates_hz_s: vec![-8., 0., 8.],
            display_bins: 32,
        };
        let threads = std::env::var("SLURM_CPUS_PER_TASK")
            .unwrap_or_else(|_| "1".into())
            .parse()?;
        let backend = Backend::Cpu;
        #[cfg(feature = "cuda")]
        let backend = if std::env::var("RUISDAP_TEST_CUDA").as_deref() == Ok("1") {
            Backend::Cuda {
                device: std::env::var("RUISDAP_CUDA_DEVICE")?.parse()?,
                memory_budget_bytes: 512 * 1024 * 1024,
                tile_ranges: 8,
            }
        } else {
            backend
        };
        let shard = cluster::search(world, &tx, &rx, &s, threads, backend)?;
        let serial = search_cpu(&tx, &rx, &s)?;
        if let Some(out) = shard.output {
            let expected = &serial.power[shard.rows.start * 32..shard.rows.end * 32];
            ensure!(out.power.len() == expected.len(), "shape mismatch");
            let scale = expected.iter().copied().fold(1_f32, f32::max);
            ensure!(
                out.power
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < scale * 1e-4),
                "MPI/serial power mismatch"
            );
            for (p, q) in out.peaks.iter().zip(&serial.peaks[shard.rows]) {
                ensure!(
                    p.delay_samples == q.delay_samples && p.doppler_hz == q.doppler_hz,
                    "peak mismatch"
                );
            }
        } else {
            ensure!(shard.rows.is_empty(), "missing rows");
        }
        // An invalid rank must make *all* ranks return an error, without hanging.
        ensure!(
            cluster::search(
                world,
                &tx,
                &rx,
                &s,
                usize::from(world.rank() != 0),
                Backend::Cpu
            )
            .is_err(),
            "collective failure did not propagate"
        );
    }
    world.barrier();
    if world.rank() == 0 {
        println!(
            "MPI serial equivalence, empty ranks and collective errors passed on {} ranks",
            world.size()
        );
    }
    Ok(())
}
fn main() -> Result<()> {
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI already initialized");
    ensure!(
        provided >= Threading::Funneled,
        "MPI lacks funneled threading"
    );
    let world = universe.world();
    if let Err(error) = run(&world) {
        eprintln!("rank {}: {error:#}", world.rank());
        world.abort(1);
    }
    Ok(())
}
