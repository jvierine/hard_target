use num_complex::Complex32 as C;
use hard_target::{search_cpu_mean_power, Settings};
fn main() -> anyhow::Result<()> {
    let n = 4096;
    let delay = 13;
    let tx: Vec<_> = (0..n)
        .map(|i| C::from_polar(1. / (n as f32).sqrt(), ((i * 73 + i * i * 17) % 997) as f32))
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
        display_bins: 127,
    };
    let cpu = search_cpu_mean_power(&tx, &rx, &s)?;
    #[cfg(feature = "cuda")]
    {
        let mut g = hard_target::gpu::GpuBank::new(0)?;
        let p = g.mean_power(&tx, &rx, &s)?;
        for (a, b) in cpu.power.iter().zip(p.power) {
            assert!(
                (*a - b).abs() <= a.abs() * 2e-4 + 1e-6,
                "CPU/GPU power bin mismatch {a} {b}"
            );
        }
        println!(
            "Pointwise GPU matched power agrees with CPU, including unequal display-bin widths"
        );
    }
    Ok(())
}
