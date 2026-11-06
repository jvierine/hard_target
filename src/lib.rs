//! Complex64 range–Doppler–Doppler-rate matched-filter bank.
//! Algorithm reference: jvierine/hard_target, commit 1b1d2b0 (MIT).
//! TX is unconjugated and explicitly masked by the caller. Positive Doppler
//! means exp(+i 2π f t) in received IQ. Delays are two-way propagation samples.
//! Coarse search holds delay fixed within a coherent interval; migration must
//! be checked/refined separately. No automatic clutter gate is inserted.
use anyhow::{ensure, Result};
use num_complex::Complex32 as C;
use rayon::prelude::*;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
pub mod gpu;
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub sample_rate_hz: f32,
    pub decimation: usize,
    pub delays: Vec<usize>,
    /// Phase correction exp(-i π rate t²), in Hz/s (no factor-of-two ambiguity).
    pub doppler_rates_hz_s: Vec<f32>,
    pub display_bins: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Peak {
    pub delay_samples: usize,
    pub doppler_hz: f32,
    pub doppler_rate_hz_s: f32,
    pub power: f32,
    pub noise_power: f32,
    pub snr_db: f32,
}
#[derive(Serialize, Deserialize)]
pub struct Output {
    pub peaks: Vec<Peak>,
    pub display_bins: usize,
    pub fft_len: usize,
    pub power: Vec<f32>,
}
pub(crate) fn validate(tx: &[C], rx: &[C], s: &Settings) -> Result<usize> {
    ensure!(
        s.sample_rate_hz.is_finite()
            && s.sample_rate_hz > 0.
            && s.decimation > 0
            && s.display_bins > 0,
        "invalid sampling/grid"
    );
    ensure!(
        !s.delays.is_empty()
            && !s.doppler_rates_hz_s.is_empty()
            && s.doppler_rates_hz_s.iter().all(|v| v.is_finite()),
        "empty/nonfinite search grid"
    );
    ensure!(
        tx.len() >= s.decimation && tx.len() % s.decimation == 0,
        "coherent interval must be divisible by decimation"
    );
    ensure!(
        s.delays
            .iter()
            .all(|&r| r.checked_add(tx.len()).is_some_and(|end| end <= rx.len())),
        "receive buffer lacks delayed support"
    );
    ensure!(
        tx.iter()
            .chain(rx)
            .all(|v| v.re.is_finite() && v.im.is_finite()),
        "mask bad input samples explicitly before search"
    );
    ensure!(tx.iter().any(|v| v.norm_sqr() > 0.), "empty TX support");
    Ok(tx.len() / s.decimation)
}
pub(crate) fn phase(s: &Settings, n: usize, rate: f32) -> Vec<C> {
    (0..n)
        .map(|i| {
            let t = (i * s.decimation) as f32 / s.sample_rate_hz;
            C::from_polar(1., -std::f32::consts::PI * rate * t * t)
        })
        .collect()
}
pub(crate) fn summarize(
    s: &Settings,
    n: usize,
    delay: usize,
    rate: f32,
    spectrum: &[C],
    best: &mut Peak,
    display: &mut [f32],
    mean_power: bool,
) {
    let mut powers: Vec<f32> = spectrum.iter().map(|v| v.norm_sqr()).collect();
    let (index, &peak) = powers
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    for (i, &p) in powers.iter().enumerate() {
        let shifted = (i + n / 2) % n;
        let bin = shifted * s.display_bins / n;
        if mean_power {
            let width =
                ((bin + 1) * n).div_ceil(s.display_bins) - (bin * n).div_ceil(s.display_bins);
            display[bin] += p / width as f32;
        } else {
            display[bin] = display[bin].max(p);
        }
    }
    if peak > best.power {
        let mid = powers.len() / 2;
        let median = *powers.select_nth_unstable_by(mid, f32::total_cmp).1;
        let noise = median / std::f32::consts::LN_2;
        let frequency_bin = if index < n.div_ceil(2) {
            index as isize
        } else {
            index as isize - n as isize
        };
        *best = Peak {
            delay_samples: delay,
            doppler_hz: frequency_bin as f32 * s.sample_rate_hz / (s.decimation * n) as f32,
            doppler_rate_hz_s: rate,
            power: peak,
            noise_power: noise,
            snr_db: if noise > 0. {
                10. * (peak / noise).log10()
            } else {
                0.
            },
        };
    }
}
pub(crate) fn empty_peak(delay: usize) -> Peak {
    Peak {
        delay_samples: delay,
        doppler_hz: 0.,
        doppler_rate_hz_s: 0.,
        power: 0.,
        noise_power: 0.,
        snr_db: 0.,
    }
}
pub fn search_cpu(tx: &[C], rx: &[C], s: &Settings) -> Result<Output> {
    search_cpu_mode(tx, rx, s, false)
}
pub fn search_cpu_mean_power(tx: &[C], rx: &[C], s: &Settings) -> Result<Output> {
    ensure!(
        s.doppler_rates_hz_s.len() == 1,
        "mean power requires one rate"
    );
    search_cpu_mode(tx, rx, s, true)
}
fn search_cpu_mode(tx: &[C], rx: &[C], s: &Settings, mean_power: bool) -> Result<Output> {
    let n = validate(tx, rx, s)?;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let phases: Vec<_> = s
        .doppler_rates_hz_s
        .iter()
        .map(|&a| phase(s, n, a))
        .collect();
    let rows: Vec<_> = s
        .delays
        .par_iter()
        .map(|&delay| {
            let echo = hard_target_signal::single::cross_products(
                &rx[delay..delay + tx.len()],
                tx,
                s.decimation,
            )
            .unwrap();
            let mut best = empty_peak(delay);
            let mut display = vec![0.; s.display_bins];
            let mut z = vec![C::default(); n];
            let mut scratch = vec![C::default(); fft.get_inplace_scratch_len()];
            for (&rate, p) in s.doppler_rates_hz_s.iter().zip(&phases) {
                for ((z, &e), &p) in z.iter_mut().zip(&echo).zip(p) {
                    *z = e * p;
                }
                fft.process_with_scratch(&mut z, &mut scratch);
                summarize(s, n, delay, rate, &z, &mut best, &mut display, mean_power);
            }
            (best, display)
        })
        .collect();
    let mut peaks = vec![];
    let mut power = vec![];
    for (p, v) in rows {
        peaks.push(p);
        power.extend(v);
    }
    Ok(Output {
        peaks,
        power,
        display_bins: s.display_bins,
        fft_len: n,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delay_frequency_and_rate() {
        let n = 2048;
        let fs = 2048.;
        let delay = 19;
        let fd = -63.;
        let rate = 16.;
        let tx: Vec<_> = (0..n)
            .map(|i| C::from_polar(1., ((i * 73 + i * i * 17) % 997) as f32))
            .collect();
        let mut rx = vec![C::default(); n + 40];
        for i in 0..n {
            let t = i as f32 / fs;
            rx[i + delay] = tx[i]
                * C::from_polar(
                    1.,
                    2. * std::f32::consts::PI * fd * t + std::f32::consts::PI * rate * t * t,
                );
        }
        let s = Settings {
            sample_rate_hz: fs,
            decimation: 1,
            delays: (0..40).collect(),
            doppler_rates_hz_s: vec![0., 16., 32.],
            display_bins: 128,
        };
        let y = search_cpu(&tx, &rx, &s).unwrap();
        let p = y
            .peaks
            .iter()
            .max_by(|a, b| a.power.total_cmp(&b.power))
            .unwrap();
        assert_eq!(p.delay_samples, delay);
        assert_eq!(p.doppler_hz, fd);
        assert_eq!(p.doppler_rate_hz_s, rate);
        assert!((p.power / (n * n) as f32 - 1.).abs() < 1e-5);
    }
}

/// Cell-averaging spectral CFAR. pfa is nominal per CUT under independent
/// exponential training cells, not a global or waveform-calibrated PFA.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CfarSettings {
    pub range_training_each_side: usize,
    pub range_guard_cells: usize,
    pub training_each_side: usize,
    pub guard_cells: usize,
    pub pfa: f32,
    pub minimum_snr_db: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Detection {
    pub delay_samples: usize,
    pub doppler_hz: f32,
    pub doppler_rate_hz_s: f32,
    pub power: f32,
    pub noise_power: f32,
    pub snr_db: f32,
}
impl CfarSettings {
    pub fn threshold(&self, n: usize) -> Result<f32> {
        ensure!(
            self.training_each_side > 0
                && self.guard_cells > 0
                && 2 * (self.training_each_side + self.guard_cells) + 1 < n,
            "CFAR stencil does not fit FFT"
        );
        ensure!(
            self.pfa > 0. && self.pfa < 1. && self.minimum_snr_db.is_finite(),
            "invalid CFAR probability/threshold"
        );
        let cells = (2 * self.training_each_side) as f32;
        Ok((cells * (self.pfa.powf(-1. / cells) - 1.)).max(10_f32.powf(self.minimum_snr_db / 10.)))
    }
}
pub(crate) fn detection(
    s: &Settings,
    n: usize,
    delay: usize,
    rate: f32,
    index: usize,
    p: f32,
    noise: f32,
) -> Detection {
    let k = if index < n.div_ceil(2) {
        index as isize
    } else {
        index as isize - n as isize
    };
    Detection {
        delay_samples: delay,
        doppler_hz: k as f32 * s.sample_rate_hz / (s.decimation * n) as f32,
        doppler_rate_hz_s: rate,
        power: p,
        noise_power: noise,
        snr_db: 10. * (p / noise).log10(),
    }
}
pub fn cfar_spectrum(
    s: &Settings,
    cfar: &CfarSettings,
    delay: usize,
    rate: f32,
    z: &[C],
) -> Result<Vec<Detection>> {
    let n = z.len();
    let threshold = cfar.threshold(n)?;
    let p: Vec<_> = z.iter().map(|x| x.norm_sqr()).collect();
    let mut hits = vec![];
    for i in 0..n {
        if p[i] <= p[(i + n - 1) % n] || p[i] < p[(i + 1) % n] {
            continue;
        }
        let mut sum = 0_f64;
        for j in 1..=cfar.training_each_side {
            let offset = cfar.guard_cells + j;
            sum += p[(i + n - offset) % n] as f64 + p[(i + offset) % n] as f64;
        }
        let noise = (sum / (2 * cfar.training_each_side) as f64) as f32;
        if noise > 0. && p[i] > threshold * noise {
            hits.push(detection(s, n, delay, rate, i, p[i], noise));
        }
    }
    Ok(hits)
}
pub fn detect_cpu(tx: &[C], rx: &[C], s: &Settings, cfar: &CfarSettings) -> Result<Vec<Detection>> {
    let n = validate(tx, rx, s)?;
    cfar.threshold(n)?;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let echoes: Vec<_> = s
        .delays
        .par_iter()
        .map(|&delay| {
            hard_target_signal::single::cross_products(&rx[delay..delay + tx.len()], tx, s.decimation)
                .unwrap()
        })
        .collect();
    let mut all = vec![];
    for &rate in &s.doppler_rates_hz_s {
        let p = phase(s, n, rate);
        let rows: Vec<Vec<C>> = echoes
            .par_iter()
            .map(|e| {
                let mut z: Vec<_> = e.iter().zip(&p).map(|(a, b)| a * b).collect();
                fft.process(&mut z);
                z
            })
            .collect();
        let hits: Vec<Result<Vec<Detection>>> = rows
            .par_iter()
            .enumerate()
            .map(|(row, z)| {
                let mut hits = cfar_spectrum(s, cfar, s.delays[row], rate, z)?;
                if cfar.range_training_each_side > 0 {
                    hits.retain_mut(|hit| {
                        let bin = (hit.doppler_hz * s.decimation as f32 * n as f32
                            / s.sample_rate_hz)
                            .round() as isize;
                        let bin = bin.rem_euclid(n as isize) as usize;
                        let mut sum = 0_f64;
                        let mut count = 0;
                        for j in 1..=cfar.range_training_each_side {
                            let d = cfar.range_guard_cells + j;
                            if row >= d {
                                sum += rows[row - d][bin].norm_sqr() as f64;
                                count += 1;
                            }
                            if row + d < rows.len() {
                                sum += rows[row + d][bin].norm_sqr() as f64;
                                count += 1;
                            }
                        }
                        if count < 4 {
                            return false;
                        }
                        let noise = (sum / count as f64) as f32;
                        let alpha = (count as f32 * (cfar.pfa.powf(-1. / count as f32) - 1.))
                            .max(10_f32.powf(cfar.minimum_snr_db / 10.));
                        if noise <= 0. || hit.power <= alpha * noise {
                            return false;
                        }
                        hit.noise_power = hit.noise_power.max(noise);
                        hit.snr_db = 10. * (hit.power / hit.noise_power).log10();
                        true
                    });
                }
                Ok(hits)
            })
            .collect();
        for h in hits {
            all.extend(h?);
        }
    }
    Ok(all)
}
#[cfg(test)]
mod cfar_tests {
    use super::*;
    #[test]
    fn local_background_and_line() {
        let n = 512;
        let mut z = vec![C::new(1., 0.); n];
        z[111] = C::new(20., 0.);
        z[333] = C::new(5., 0.);
        let s = Settings {
            sample_rate_hz: 512.,
            decimation: 1,
            delays: vec![7],
            doppler_rates_hz_s: vec![0.],
            display_bins: 16,
        };
        let c = CfarSettings {
            range_training_each_side: 4,
            range_guard_cells: 2,
            training_each_side: 32,
            guard_cells: 3,
            pfa: 1e-6,
            minimum_snr_db: 12.,
        };
        let y = cfar_spectrum(&s, &c, 7, 0., &z).unwrap();
        assert_eq!(y.len(), 2);
        assert_eq!(y[0].doppler_hz, 111.);
        assert_eq!(y[1].doppler_hz, -179.);
        let base = cfar_spectrum(&s, &c, 7, 0., &vec![C::new(3., 0.); n]).unwrap();
        assert!(base.is_empty());
    }
}

#[cfg(feature = "arecibo")]
pub mod arecibo;

#[cfg(feature = "mpi")]
pub mod cluster;
