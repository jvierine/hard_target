//! Arecibo measured-waveform products from the same staged legacy recording.
use anyhow::{ensure, Context, Result};
use guisdap_digital_rf::legacy::LegacyFile;
use num_complex::{Complex32 as C, Complex64 as D};
use rayon::prelude::*;
use std::{fs::File, io::Write, path::Path};
#[derive(Clone)]
pub struct Recording {
    pub iq: Vec<C>,
    pub start_sample: u64,
    pub sample_rate_hz: u64,
    pub tx: [usize; 2],
    pub ipp: usize,
}
impl Recording {
    pub fn open(path: &Path) -> Result<Self> {
        let file = LegacyFile::open(path)?;
        let mut blocks = file.read_rows(0..file.rows())?;
        ensure!(
            file.sample_rate_hz == 25_000_000 && blocks.len() == 1,
            "unsupported Arecibo layout/gap"
        );
        let block = blocks.pop().unwrap();
        let ipp = 250000;
        let mut envelope = vec![0_f64; ipp / 250];
        let pulses = block.samples.len() / ipp;
        for pulse in block.samples.chunks_exact(ipp) {
            for (j, x) in pulse.chunks_exact(250).enumerate() {
                envelope[j] += x.iter().map(|z| z.norm_sqr() as f64).sum::<f64>();
            }
        }
        ensure!(pulses > 10, "short recording");
        let peak = envelope
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let threshold = envelope[peak] * 0.4;
        let (mut lo, mut hi) = (peak, peak);
        while lo > 0 && envelope[lo - 1] > threshold {
            lo -= 1
        }
        while hi + 1 < envelope.len() && envelope[hi + 1] > threshold {
            hi += 1
        }
        let tx = [lo.saturating_sub(1) * 250, ((hi + 2) * 250).min(ipp)];
        ensure!(
            (10000..13000).contains(&(tx[1] - tx[0])),
            "TX not compatible with 440 us campaign"
        );
        Ok(Self {
            iq: block.samples,
            start_sample: block.start_sample,
            sample_rate_hz: file.sample_rate_hz,
            tx,
            ipp,
        })
    }
    pub fn time(&self) -> f64 {
        self.start_sample as f64 / self.sample_rate_hz as f64
    }
}
pub fn plasma(r: &Recording, dir: &Path, gpu: bool) -> Result<()> {
    use crate::{search_cpu_mean_power, Settings};
    std::fs::create_dir_all(dir)?;
    let fs = r.sample_rate_hz as f32;
    let txlen = r.tx[1] - r.tx[0];
    let ranges: Vec<f32> = (0..141).map(|i| 50_000. + i as f32 * 7500.).collect();
    let delays: Vec<_> = ranges
        .iter()
        .map(|x| (2. * x / 299792458. * fs).round() as usize)
        .collect();
    let end = *delays.last().unwrap() + txlen;
    let settings = Settings {
        sample_rate_hz: fs,
        decimation: 1,
        delays,
        doppler_rates_hz_s: vec![0.],
        display_bins: 2048,
    };
    #[cfg(feature = "cuda")]
    let mut engine = if gpu {
        Some(crate::gpu::GpuBank::new(0)?)
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    ensure!(!gpu, "CUDA not compiled");
    let rails: Vec<usize> =
        r.iq.iter()
            .enumerate()
            .filter_map(|(i, z)| {
                if z.re.abs() >= 32767. || z.im.abs() >= 32767. {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
    let mut sum = vec![0_f64; ranges.len() * settings.display_bins];
    let mut squares = sum.clone();
    let mut count = 0;
    for start in (r.tx[0]..r.iq.len().saturating_sub(end)).step_by(r.ipp) {
        let mut tx = r.iq[start..start + txlen].to_vec();
        let energy = tx.iter().map(|z| z.norm_sqr() as f64).sum::<f64>().sqrt() as f32;
        ensure!(
            energy > 0. && energy.is_finite(),
            "invalid measured TX energy"
        );
        for x in &mut tx {
            *x /= energy;
        }
        let mut rx = r.iq[start..start + end].to_vec();
        for (i, x) in rx.iter_mut().enumerate() {
            let gate = (start + i) % r.ipp;
            if (r.tx[0]..r.tx[1]).contains(&gate) || x.re.abs() >= 32767. || x.im.abs() >= 32767. {
                *x = C::default();
            }
        }
        #[cfg(feature = "cuda")]
        let p = if let Some(g) = engine.as_mut() {
            g.mean_power(&tx, &rx, &settings)?
        } else {
            search_cpu_mean_power(&tx, &rx, &settings)?
        };
        #[cfg(not(feature = "cuda"))]
        let p = search_cpu_mean_power(&tx, &rx, &settings)?;
        let mut prefix = vec![0_f64; txlen + 1];
        for i in 0..txlen {
            prefix[i + 1] = prefix[i] + tx[i].norm_sqr() as f64;
        }
        for (row, power) in p.power.chunks_exact(settings.display_bins).enumerate() {
            let delay = settings.delays[row];
            let mut weight = prefix[txlen];
            if delay < txlen {
                weight -= prefix[txlen - delay];
            }
            let begin = start + delay;
            let end = begin + txlen;
            let lo = rails.partition_point(|&i| i < begin);
            let hi = rails.partition_point(|&i| i < end);
            for &i in &rails[lo..hi] {
                let k = i - begin;
                let phase = i % r.ipp;
                if !(r.tx[0]..r.tx[1]).contains(&phase) {
                    weight -= tx[k].norm_sqr() as f64;
                }
            }
            ensure!(
                weight > 0.05,
                "insufficient unmasked plasma receive support at range {}",
                ranges[row]
            );
            for (col, &p) in power.iter().enumerate() {
                let i = row * settings.display_bins + col;
                let v = p as f64 / fs as f64 / weight;
                sum[i] += v;
                squares[i] += v * v;
            }
        }
        count += 1;
    }
    ensure!(count > 1, "insufficient plasma-line pulses");
    let mean: Vec<f32> = sum.iter().map(|s| (s / count as f64) as f32).collect();
    let error: Vec<f32> = sum
        .iter()
        .zip(squares)
        .map(|(s, q)| {
            ((q - s * s / count as f64).max(0.) / (count * (count - 1)) as f64).sqrt() as f32
        })
        .collect();
    let frequency: Vec<f32> = (0..settings.display_bins)
        .map(|i| -fs / 2. + (i as f32 + 0.5) * fs / settings.display_bins as f32)
        .collect();
    let out = hdf5::File::create(dir.join("plasma-line.h5"))?;
    out.new_dataset_builder()
        .with_data(&frequency)
        .create("frequency_offset_hz")?;
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_m")?;
    for (name, v) in [
        ("matched_psd_adu2_hz", &mean),
        ("standard_error_adu2_hz", &error),
    ] {
        out.new_dataset::<f32>()
            .shape((ranges.len(), settings.display_bins))
            .deflate(4)
            .create(name)?
            .write_raw(v)?;
    }
    let meta = serde_json::json!({"schema":"ruisdap-arecibo-plasma-rd-v1","start_unix_s":r.time(),"duration_s":r.iq.len()as f64/r.sample_rate_hz as f64,"pulse_count":count,"tx_window_samples":r.tx,"frequency_bins":settings.display_bins,"frequency_min_hz":-fs/2.,"frequency_max_hz":fs/2.,"range_m":ranges,"power_file":"power.f32","range_bins":ranges.len(),"precision":"complex64 FFT; float64 sums; float32 product","units":"ADU squared per Hz, measured-TX unit-energy matched periodogram","frequency_sign":"Recorded IQ Doppler relative to measured transmitter; receiver LO cancels in matched product","range_resolution":"Measured pulse ambiguity; output sampling 7.5 km does not imply 7.5 km physical resolution","receiver_background_subtracted":false,"absolute_density_calibrated":false,"backend":if gpu{"gpu"}else{"cpu"}});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    out.flush()?;
    drop(out);
    let mut f = File::create(dir.join("power.f32"))?;
    for v in mean {
        f.write_all(&v.to_le_bytes())?;
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec(&meta)?)?;
    plasma_highpass(dir)?;
    println!("Plasma RD: {count} pulses, {} ranges", ranges.len());
    Ok(())
}
/// Robust spectral high-pass, independently at each slant range. This removes
/// a locally smooth receiver pedestal, not the pulse ambiguity. Signed residuals
/// are retained. Broad physical spectra can be attenuated by this diagnostic.
pub fn spectral_highpass(
    data: &[f32],
    bins: usize,
    half: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    ensure!(
        bins > 2 * half && half > 0 && data.len() % bins == 0,
        "invalid high-pass shape/window"
    );
    ensure!(
        data.iter().all(|v| v.is_finite() && *v >= 0.),
        "nonfinite/negative periodogram input"
    );
    let rows: Vec<_> = data
        .par_chunks_exact(bins)
        .map(|row| {
            let mut background = Vec::with_capacity(bins);
            let mut residual = Vec::with_capacity(bins);
            let mut contrast = Vec::with_capacity(bins);
            let mut window = vec![0.; 2 * half + 1];
            for i in 0..bins {
                for (j, v) in window.iter_mut().enumerate() {
                    // Reflect at Nyquist edges; do not wrap opposite receiver edges.
                    let k = i as isize + j as isize - half as isize;
                    let k = if k < 0 {
                        -k
                    } else if k >= bins as isize {
                        2 * bins as isize - 2 - k
                    } else {
                        k
                    };
                    *v = row[k as usize];
                }
                let b = *window.select_nth_unstable_by(half, f32::total_cmp).1;
                let y = row[i] - b;
                background.push(b);
                residual.push(y);
                contrast.push(if b > 0. { y / b } else { f32::NAN });
            }
            (background, residual, contrast)
        })
        .collect();
    let mut background = Vec::with_capacity(data.len());
    let mut residual = Vec::with_capacity(data.len());
    let mut contrast = Vec::with_capacity(data.len());
    for (b, r, c) in rows {
        background.extend(b);
        residual.extend(r);
        contrast.extend(c);
    }
    Ok((background, residual, contrast))
}
/// Add compact diagnostic products to a new or existing matched-PSD chunk.
/// Safe to rerun; the original measured spectrum and its errors are unchanged.
pub fn plasma_highpass(dir: &Path) -> Result<()> {
    let file = hdf5::File::open_rw(dir.join("plasma-line.h5"))?;
    let ds = file.dataset("matched_psd_adu2_hz")?;
    let shape = ds.shape();
    ensure!(shape.len() == 2, "invalid matched PSD shape");
    let data: Vec<f32> = ds.read_raw()?;
    let bins = shape[1];
    let frequency: Vec<f32> = file.dataset("frequency_offset_hz")?.read_raw()?;
    ensure!(
        frequency.len() == bins && bins > 1,
        "invalid frequency axis"
    );
    // About 0.5 MHz full median window at this output sampling. The exact
    // sampled width is stored, never presented as deconvolved resolution.
    let df = frequency[1] - frequency[0];
    ensure!(df > 0., "frequency axis must increase");
    let half = ((250_000. / df).round() as usize)
        .max(1)
        .min((bins - 1) / 2);
    let (background, residual, contrast) = spectral_highpass(&data, bins, half)?;
    for (name, v, export) in [
        ("spectral_background_adu2_hz", &background, "background.f32"),
        ("highpass_psd_adu2_hz", &residual, "highpass.f32"),
        ("highpass_fractional_excess", &contrast, "contrast.f32"),
    ] {
        if file.link_exists(name) {
            file.unlink(name)?;
        }
        file.new_dataset::<f32>()
            .shape(shape.clone())
            .deflate(4)
            .create(name)?
            .write_raw(v)?;
        let mut f = File::create(dir.join(export))?;
        for x in v {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    meta["highpass"] = serde_json::json!({
        "revision":"spectral-median-v1", "method":"per-range reflected running spectral median",
        "window_bins":2*half+1, "window_hz":(2*half+1) as f32*df,
        "background_file":"background.f32", "residual_file":"highpass.f32", "contrast_file":"contrast.f32",
        "signed":true, "range_doppler_deconvolved":false,
        "contrast_definition":"(matched PSD - local spectral median) / local spectral median; not a detection significance",
        "limitations":"Suppresses features broad compared with the median window; background uncertainty is not included in the original standard error. Pulse range-Doppler ambiguity remains."
    });
    if file.link_exists("metadata_json_utf8") {
        file.unlink("metadata_json_utf8")?;
    }
    let bytes = serde_json::to_vec(&meta)?;
    file.new_dataset_builder()
        .with_data(&bytes)
        .create("metadata_json_utf8")?;
    file.flush()?;
    std::fs::write(dir.join("manifest.json"), bytes)?;
    Ok(())
}
#[cfg(test)]
mod highpass_tests {
    use super::*;
    #[test]
    fn pedestal_and_signed_line_preservation() {
        let mut data = vec![10.; 202];
        data[40] = 15.;
        data[65] = 7.;
        data[101 + 40] = 25.;
        let (b, r, c) = spectral_highpass(&data, 101, 10).unwrap();
        assert!(b.iter().all(|x| *x == 10.));
        assert_eq!(r[40], 5.);
        assert_eq!(r[65], -3.);
        assert_eq!(c[40], 0.5);
        assert_eq!(c[65], -0.3);
        assert_eq!(r[101 + 40], 15.); // no mixing between ranges
        assert_eq!(r[0], 0.);
        assert_eq!(r[100], 0.);
    }
    #[test]
    fn hdf5_postprocess_is_signed_idempotent_and_preserves_measurement() {
        let dir =
            std::env::temp_dir().join(format!("ruisdap-highpass-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut data = vec![10_f32; 202];
        data[40] = 15.;
        data[65] = 7.;
        let frequency: Vec<f32> = (0..101).map(|i| i as f32 * 25_000.).collect();
        let file = hdf5::File::create(dir.join("plasma-line.h5")).unwrap();
        file.new_dataset::<f32>()
            .shape((2, 101))
            .create("matched_psd_adu2_hz")
            .unwrap()
            .write_raw(&data)
            .unwrap();
        file.new_dataset_builder()
            .with_data(&frequency)
            .create("frequency_offset_hz")
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[2_f32; 202])
            .create("standard_error_adu2_hz")
            .unwrap();
        drop(file);
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();
        plasma_highpass(&dir).unwrap();
        plasma_highpass(&dir).unwrap();
        let file = hdf5::File::open(dir.join("plasma-line.h5")).unwrap();
        assert_eq!(
            file.dataset("matched_psd_adu2_hz")
                .unwrap()
                .read_raw::<f32>()
                .unwrap(),
            data
        );
        assert_eq!(
            file.dataset("standard_error_adu2_hz")
                .unwrap()
                .read_raw::<f32>()
                .unwrap(),
            vec![2.; 202]
        );
        let residual = file
            .dataset("highpass_psd_adu2_hz")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();
        assert_eq!(residual[40], 5.);
        assert_eq!(residual[65], -3.);
        let bytes = std::fs::read(dir.join("highpass.f32")).unwrap();
        let binary: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(binary, residual);
        let metadata: serde_json::Value = serde_json::from_slice(
            &file
                .dataset("metadata_json_utf8")
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["highpass"]["range_doppler_deconvolved"], false);
        drop(file);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn rejects_invalid_shape_and_values() {
        assert!(spectral_highpass(&[1.; 10], 3, 1).is_err());
        assert!(spectral_highpass(&[f32::NAN; 12], 6, 1).is_err());
    }
}
fn fir(x: &[C], taps: &[f32], decimation: usize) -> Vec<C> {
    let half = taps.len() / 2;
    (0..x.len() / decimation)
        .into_par_iter()
        .map(|i| {
            let center = i * decimation;
            if center < half || center + half >= x.len() {
                return C::default();
            }
            let mut z = C::default();
            for (j, &a) in taps.iter().enumerate() {
                z += x[center - half + j] * a;
            }
            z
        })
        .collect()
}
/// Experimental WLS LPI retaining actual pulse/receiver-mask response. No SVD.
fn contamination_candidates(cfar: &[serde_json::Value]) -> Vec<&serde_json::Value> {
    cfar.iter()
        .filter(|d| {
            let snr = d["snr_db"].as_f64().unwrap_or(0.);
            if snr >= 35. {
                return true;
            }
            if snr < 20. {
                return false;
            }
            let time = d["pulse_time_unix_s"].as_f64().unwrap_or(0.);
            let range = d["range_m"].as_f64().unwrap_or(0.);
            let frequency = d["doppler_hz"].as_f64().unwrap_or(0.);
            let mut times = vec![];
            for other in cfar {
                let t = other["pulse_time_unix_s"].as_f64().unwrap_or(0.);
                if other["snr_db"].as_f64().unwrap_or(0.) >= 20.
                    && (t - time).abs() <= 0.25
                    && (other["range_m"].as_f64().unwrap_or(0.) - range).abs() <= 20_000.
                    && (other["doppler_hz"].as_f64().unwrap_or(0.) - frequency).abs() <= 2500.
                    && !times.iter().any(|x: &f64| (*x - t).abs() < 1e-5)
                {
                    times.push(t);
                }
            }
            times.len() >= 3
        })
        .collect()
}
pub fn ion_lpi(r: &Recording, dir: &Path, cfar: &[serde_json::Value]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let decimation = 250;
    let dt = decimation as f64 / r.sample_rate_hz as f64;
    let half = 1000;
    let fs = r.sample_rate_hz as f64;
    let cutoff = 30_000.;
    let mut taps: Vec<f32> = (0..=2 * half)
        .map(|i| {
            let x = i as f64 - half as f64;
            let sinc = if x == 0. {
                2. * cutoff / fs
            } else {
                (2. * std::f64::consts::PI * cutoff * x / fs).sin() / (std::f64::consts::PI * x)
            };
            (sinc
                * (0.54 - 0.46 * (2. * std::f64::consts::PI * i as f64 / (2 * half) as f64).cos()))
                as f32
        })
        .collect();
    let norm = taps.iter().sum::<f32>();
    for t in &mut taps {
        *t /= norm;
    }
    let echo = fir(&r.iq, &taps, decimation);
    let ipp = r.ipp / decimation;
    let tx0 = r.tx[0] / decimation;
    let width = (r.tx[1] - r.tx[0]) / decimation;
    let pad = half / decimation;
    let support = width + 2 * pad;
    let pulses = (r.iq.len() / r.ipp).saturating_sub(1);
    let mut tx = Vec::new();
    for p in 0..pulses {
        let mut w = vec![C::default(); (support + 2 * pad) * decimation];
        let at = p * r.ipp + r.tx[0];
        let begin = 2 * pad * decimation;
        for i in 0..r.tx[1] - r.tx[0] {
            w[begin + i] = r.iq[at + i];
        }
        let f = fir(&w, &taps, decimation);
        tx.push(f[pad..pad + support].to_vec());
    }
    let energy = tx
        .iter()
        .flat_map(|v| v.iter())
        .map(|z| z.norm_sqr() as f64)
        .sum::<f64>()
        / pulses as f64;
    ensure!(energy > 0., "no TX energy");
    let scale = energy.sqrt() as f32;
    for v in &mut tx {
        for z in v {
            *z /= scale;
        }
    }
    // Estimate wideband white receiver noise from first differences outside TX.
    // Stored as a provisional assumption, never an absolute density calibration.
    let mut powers = Vec::new();
    for chunk in r.iq.chunks_exact(4096).enumerate() {
        let (i, x) = chunk;
        let center = i * 4096 + 2048;
        let phase = center % r.ipp;
        if phase >= r.tx[0].saturating_sub(4096) && phase < r.tx[1] + 4096 {
            continue;
        }
        let v = x
            .windows(2)
            .map(|v| (v[1] - v[0]).norm_sqr() as f64 / 2.)
            .sum::<f64>()
            / (x.len() - 1) as f64;
        powers.push(v);
    }
    ensure!(!powers.is_empty(), "no receiver-noise samples");
    powers.sort_by(f64::total_cmp);
    let noise = powers[powers.len() / 2];
    let lags = 41;
    let columns = 201;
    let range_step = 5usize;
    let range_m: Vec<f32> = (0..columns)
        .map(|i| (i * range_step) as f64 * dt * 299792458. / 2.)
        .map(|v| v as f32)
        .collect();
    let mut usable = vec![true; echo.len()];
    for (i, u) in usable.iter_mut().enumerate() {
        let k = i * decimation;
        let phase = k % r.ipp;
        if k < half || k + half >= r.iq.len() || (phase + half >= r.tx[0] && phase < r.tx[1] + half)
        {
            *u = false;
        }
    }
    for (i, z) in r.iq.iter().enumerate() {
        if z.re.abs() >= 32767. || z.im.abs() >= 32767. {
            let lo = i.saturating_sub(half) / decimation;
            let hi = ((i + half) / decimation + 1).min(usable.len());
            for u in &mut usable[lo..hi] {
                *u = false;
            }
        }
    }
    let screened = contamination_candidates(cfar);
    for d in &screened {
        if d["doppler_hz"].as_f64().unwrap_or(0.).abs() > 60_000.
            && d["snr_db"].as_f64().unwrap_or(0.) < 50.
        {
            continue;
        }
        if let (Some(t), Some(range)) = (d["pulse_time_unix_s"].as_f64(), d["range_m"].as_f64()) {
            let count = d["pulse_count"].as_u64().unwrap_or(1) as usize;
            for p in 0..count {
                let lo = (((t - r.time()) + p as f64 * r.ipp as f64 / fs + 2. * range / 299792458.)
                    / dt)
                    .floor() as isize
                    - pad as isize;
                let hi = lo + support as isize + 2 * pad as isize;
                for k in lo.max(0) as usize..(hi.max(0) as usize).min(usable.len()) {
                    usable[k] = false;
                }
            }
        }
    }
    // Group pulses only if measured transmit shape is essentially identical up to
    // phase. Each retained row still uses the mean response of its actual pulses.
    let mut groups: Vec<Vec<usize>> = vec![];
    for p in 0..pulses {
        let mut found = None;
        for (g, members) in groups.iter().enumerate() {
            let v = &tx[members[0]];
            let cross: D = v
                .iter()
                .zip(&tx[p])
                .map(|(a, b)| D::new(a.re as f64, a.im as f64) * D::new(b.re as f64, -b.im as f64))
                .sum();
            let ea = v.iter().map(|z| z.norm_sqr() as f64).sum::<f64>();
            let eb = tx[p].iter().map(|z| z.norm_sqr() as f64).sum::<f64>();
            if cross.norm_sqr() > 0.995 * ea * eb {
                found = Some(g);
                break;
            }
        }
        if let Some(g) = found {
            groups[g].push(p)
        } else {
            groups.push(vec![p])
        }
    }
    let results: Vec<_> = (0..lags)
        .into_par_iter()
        .map(|lag| -> Result<(Vec<D>, Vec<f64>, f64, usize)> {
            let shift = lag * decimation;
            let background = if shift < taps.len() {
                noise
                    * taps[..taps.len() - shift]
                        .iter()
                        .zip(&taps[shift..])
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum::<f64>()
            } else {
                0.
            };
            let mut rows = Vec::new();
            let mut y = Vec::new();
            let mut sigma = Vec::new();
            for members in &groups {
                for t in width + pad..ipp - pad - lag {
                    let accepted: Vec<_> = members
                        .iter()
                        .copied()
                        .filter(|&p| {
                            let i = p * ipp + tx0 + t;
                            i + lag < echo.len() && usable[i] && usable[i + lag]
                        })
                        .collect();
                    if accepted.is_empty() {
                        continue;
                    }
                    let observed: Vec<D> = accepted
                        .iter()
                        .map(|&p| {
                            let i = p * ipp + tx0 + t;
                            let z = echo[i] * echo[i + lag].conj();
                            D::new(z.re as f64, z.im as f64)
                        })
                        .collect();
                    let mean = observed.iter().copied().sum::<D>() / observed.len() as f64;
                    let empirical = if observed.len() > 1 {
                        observed.iter().map(|z| (*z - mean).norm_sqr()).sum::<f64>()
                            / (observed.len() - 1) as f64
                    } else {
                        let i = accepted[0] * ipp + tx0 + t;
                        (echo[i].norm_sqr() as f64) * (echo[i + lag].norm_sqr() as f64)
                    };
                    let mut response = vec![D::default(); support];
                    for &p in &accepted {
                        for j in 0..support.saturating_sub(lag) {
                            let z = tx[p][j] * tx[p][j + lag].conj();
                            response[j] += D::new(z.re as f64, z.im as f64) / accepted.len() as f64;
                        }
                    }
                    let mut row = Vec::new();
                    for col in 0..columns {
                        let center = (col * range_step) as isize;
                        let mut w = D::default();
                        for offset in -2..=2 {
                            let j = t as isize - center - offset + pad as isize;
                            if j >= 0 && (j as usize) + lag < support {
                                w += response[j as usize];
                            }
                        }
                        if w.norm_sqr() > 1e-22 {
                            row.push((col, w));
                        }
                    }
                    if !row.is_empty() {
                        rows.push(row);
                        y.push(mean - D::new(background, 0.));
                        sigma.push(
                            (empirical / observed.len() as f64)
                                .max(noise * noise * 1e-12)
                                .sqrt(),
                        );
                    }
                }
            }
            let count = rows.len();
            let (x, v, c) = guisdap_millstone::lpi::solve(&rows, &y, &sigma, columns)
                .context("Arecibo measured-waveform lag-profile inversion")?;
            Ok((x, v, c, count))
        })
        .collect();
    let mut real = vec![f32::NAN; lags * columns];
    let mut imag = real.clone();
    let mut variance = real.clone();
    let mut status = vec![];
    let mut success = 0;
    for (lag, result) in results.into_iter().enumerate() {
        match result {
            Ok((x, v, c, n)) => {
                for i in 0..columns {
                    real[lag * columns + i] = x[i].re as f32;
                    imag[lag * columns + i] = x[i].im as f32;
                    variance[lag * columns + i] = v[i] as f32;
                }
                status.push(
                    serde_json::json!({"lag":lag,"condition_proxy":c,"rows":n,"status":"solved"}),
                );
                success += 1
            }
            Err(e) => {
                status.push(serde_json::json!({"lag":lag,"status":"failed","error":e.to_string()}))
            }
        }
    }
    ensure!(success > 0, "all measured-waveform LPI systems failed");
    let out = hdf5::File::create(dir.join("ion-lpi.h5"))?;
    out.new_dataset_builder()
        .with_data(&range_m)
        .create("slant_range_m")?;
    out.new_dataset_builder()
        .with_data(&(0..lags).map(|i| i as f64 * dt).collect::<Vec<_>>())
        .create("lag_s")?;
    for (name, v) in [
        ("acf_real_adu2", &real),
        ("acf_imag_adu2", &imag),
        ("variance_adu4", &variance),
    ] {
        out.new_dataset::<f32>()
            .shape((lags, columns))
            .deflate(4)
            .create(name)?
            .write_raw(v)?;
    }
    out.new_dataset_builder()
        .with_data(&taps)
        .create("receiver_fir_taps")?;
    out.new_attr::<u64>()
        .create("source_start_sample")?
        .write_scalar(&r.start_sample)?;
    out.new_attr::<u64>()
        .create("source_sample_rate_hz")?
        .write_scalar(&r.sample_rate_hz)?;
    let mask: Vec<u8> = usable.iter().map(|&x| u8::from(x)).collect();
    out.new_dataset_builder()
        .with_data(&mask)
        .deflate(4)
        .create("receiver_usable_mask")?;
    for (name, values) in [
        (
            "filtered_iq_real_adu",
            echo.iter().map(|z| z.re).collect::<Vec<_>>(),
        ),
        ("filtered_iq_imag_adu", echo.iter().map(|z| z.im).collect()),
    ] {
        out.new_dataset_builder()
            .with_data(&values)
            .deflate(4)
            .create(name)?;
    }
    for (name, values) in [
        (
            "measured_tx_real_normalized",
            tx.iter().flatten().map(|z| z.re).collect::<Vec<_>>(),
        ),
        (
            "measured_tx_imag_normalized",
            tx.iter().flatten().map(|z| z.im).collect(),
        ),
    ] {
        out.new_dataset::<f32>()
            .shape((pulses, support))
            .deflate(4)
            .create(name)?
            .write_raw(&values)?;
    }
    for (name, values) in [
        ("acf-real.f32", &real),
        ("acf-imag.f32", &imag),
        ("variance.f32", &variance),
    ] {
        let mut f = File::create(dir.join(name))?;
        for &v in values {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    let meta = serde_json::json!({"schema":"ruisdap-arecibo-measured-tx-lpi-v1","shape":[lags,columns],"lag_s":(0..lags).map(|i|i as f64*dt).collect::<Vec<_>>(),"slant_range_m":range_m,"acf_real_file":"acf-real.f32","acf_imag_file":"acf-imag.f32","variance_file":"variance.f32","start_unix_s":r.time(),"pulse_count":pulses,"tx_shape_groups":groups.len(),"sample_interval_s":dt,"range_cell_width_m":range_m[1]-range_m[0],"receiver_filter_cutoff_hz":cutoff,"cfar_input_detections":cfar.len(),"coherent_contamination_candidates":screened.len(),"contamination_policy":"ADC rail support always excluded; CFAR exclusions require >=35 dB alone or >=20 dB in three distinct pulse times within 0.25 s, 20 km and 2.5 kHz; out-of-ion-band weak hits do not mask ion products","receiver_noise_variance_raw_adu2":noise,"noise_model":"Provisional wideband-white receiver noise from median block mean first-difference power; FIR covariance propagated to every lag","calibration":"ADU squared; absolute receiver/density calibration not applied","status":"Experimental measured-transmit WLS LPI; no SVD or lag interpolation","lag_status":status,"precision":"complex64 filtering/products, complex128 WLS, float32 output","limitations":["White receiver-noise assumption requires validation against recorded noise spectrum","Finite range domain and unambiguous-range contamination require review","One-second stationarity and diagonal empirical product variances; cross-range/lag covariance not retained"]});
    out.new_dataset_builder()
        .with_data(&serde_json::to_vec(&meta)?)
        .create("metadata_json_utf8")?;
    out.flush()?;
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec(&meta)?)?;
    println!(
        "Ion LPI: {success}/{lags} lags, {} measured code groups",
        groups.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unconfirmed_noise_peaks_do_not_censor_ion_power() {
        let make = |t, s| serde_json::json!({"pulse_time_unix_s":t,"range_m":100000.,"doppler_hz":1000.,"snr_db":s});
        let lone = vec![make(1., 22.)];
        assert!(contamination_candidates(&lone).is_empty());
        let track = vec![make(1., 22.), make(1.01, 22.), make(1.02, 22.)];
        assert_eq!(contamination_candidates(&track).len(), 3);
        let strong = vec![make(1., 40.)];
        assert_eq!(contamination_candidates(&strong).len(), 1);
    }
    #[test]
    fn sparse_wls_missing_equations_recovers_signed_complex_profile() {
        let rows = vec![
            vec![(0, D::new(1., 0.))],
            vec![(0, D::new(0.5, 0.2)), (1, D::new(1., 0.))],
            vec![(1, D::new(1., 0.)), (2, D::new(0.5, -0.1))],
            vec![(2, D::new(1., 0.))],
        ];
        let truth = vec![D::new(2., -0.3), D::new(-0.5, 0.2), D::new(4., 0.1)];
        let y: Vec<_> = rows
            .iter()
            .map(|r| r.iter().map(|(i, w)| w * truth[*i]).sum())
            .collect();
        let (x, _, _) =
            guisdap_millstone::lpi::solve(&rows, &y, &vec![0.1; rows.len()], 3).unwrap();
        for (a, b) in x.iter().zip(&truth) {
            assert!((*a - *b).norm() < 1e-10)
        }
    }
}
