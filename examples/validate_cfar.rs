use num_complex::Complex32 as C;
use hard_target::{detect_cpu, CfarSettings, Settings};
fn main() -> anyhow::Result<()> {
    let n = 4096;
    let delay = 13;
    let tx: Vec<_> = (0..n)
        .map(|i| C::from_polar(1., ((i * 73 + i * i * 17) % 997) as f32))
        .collect();
    let mut rx = vec![C::default(); n + 30];
    for i in 0..n {
        rx[i + delay] =
            tx[i] * C::from_polar(1., 2. * std::f32::consts::PI * (-77.) * i as f32 / n as f32);
    }
    let s = Settings {
        sample_rate_hz: n as f32,
        decimation: 1,
        delays: (0..30).collect(),
        doppler_rates_hz_s: vec![0.],
        display_bins: 16,
    };
    let c = CfarSettings {
        range_training_each_side: 4,
        range_guard_cells: 2,
        training_each_side: 64,
        guard_cells: 4,
        pfa: 1e-6,
        minimum_snr_db: 12.,
    };
    let cpu = detect_cpu(&tx, &rx, &s, &c)?;
    assert!(cpu
        .iter()
        .any(|d| d.delay_samples == delay && d.doppler_hz == -77.));
    #[cfg(feature = "cuda")]
    {
        let mut g = hard_target::gpu::GpuBank::new(0)?;
        let hits = g.detect(&tx, &rx, &s, &c)?;
        assert!(hits
            .iter()
            .any(|d| d.delay_samples == delay && d.doppler_hz == -77.));
        println!(
            "CFAR CPU {} hits, GPU {} hits; injected range/Doppler recovered",
            cpu.len(),
            hits.len()
        );
    }
    Ok(())
}
