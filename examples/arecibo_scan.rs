//! Initial event search in a one-second Arecibo legacy file. No plasma fits.
use anyhow::{ensure, Result};
use guisdap_digital_rf::legacy::LegacyFile;
use num_complex::Complex32 as C;
use hard_target::{detect_cpu, CfarSettings, Settings};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "input.h5 output-directory cpu|gpu meteor|space"
    );
    let source = LegacyFile::open(&args[1])?;
    let blocks = source.read_rows(0..source.rows())?;
    ensure!(
        blocks.len() == 1 && source.sample_rate_hz == 25_000_000,
        "unsupported recording or gap"
    );
    let b = &blocks[0];
    let z = &b.samples;
    let fs = source.sample_rate_hz as f32;
    let ipp = 250000;
    ensure!(z.len() >= ipp * 10, "need at least ten pulses");
    // Fold envelope over the file: repeated TX survives, isolated echoes do not.
    let mut envelope = vec![0_f32; ipp / 250];
    let np = z.len() / ipp;
    for pulse in z.chunks_exact(ipp) {
        for (j, v) in pulse.chunks_exact(250).enumerate() {
            envelope[j] += v.iter().map(|z| z.norm_sqr()).sum::<f32>() / (250 * np) as f32;
        }
    }
    let peak = envelope
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let mut a = peak;
    let mut e = peak;
    let threshold = envelope[peak] * 0.4;
    while a > 0 && envelope[a - 1] > threshold {
        a -= 1;
    }
    while e + 1 < envelope.len() && envelope[e + 1] > threshold {
        e += 1;
    }
    let tx0 = a.saturating_sub(1) * 250;
    let tx1 = ((e + 2) * 250).min(ipp);
    ensure!(
        (10000..13000).contains(&(tx1 - tx0)),
        "TX window not consistent with campaign's 440 us pulse: {} samples",
        tx1 - tx0
    );
    println!(
        "TX support {}..{} us, {} pulses",
        tx0 as f32 / 25.,
        tx1 as f32 / 25.,
        np
    );
    let meteor = args[4] == "meteor";
    ensure!(meteor || args[4] == "space", "unknown mode");
    let pulses = if meteor { 1 } else { 8 };
    let n = ipp * pulses;
    let ranges: Vec<f32> = if meteor {
        (0..250).map(|i| 25. + i as f32 * 1.5).collect()
    } else {
        (0..220).map(|i| 90. + i as f32 * 6.).collect()
    };
    let delays: Vec<_> = ranges
        .iter()
        .map(|r| (r * 2000. / 299792458. * fs).round() as usize)
        .collect();
    let max_delay = *delays.last().unwrap();
    let s = Settings {
        sample_rate_hz: fs,
        decimation: 50,
        delays,
        doppler_rates_hz_s: if meteor {
            vec![0.]
        } else {
            vec![-60., 0., 60.]
        },
        display_bins: 256,
    };
    let dir = Path::new(&args[2]);
    std::fs::create_dir_all(dir)?;
    #[cfg(feature = "cuda")]
    let mut gpu = if args[3] == "gpu" {
        Some(hard_target::gpu::GpuBank::new(0)?)
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    ensure!(args[3] == "cpu", "CUDA feature not compiled");
    ensure!(args[3] == "cpu" || args[3] == "gpu", "unknown backend");
    let fft_len = n / s.decimation;
    let cfar = CfarSettings {
        range_training_each_side: 8,
        range_guard_cells: if meteor { 44 } else { 11 },
        training_each_side: 64,
        guard_cells: ((3. * fs / (tx1 - tx0) as f32) / (fs / n as f32)).ceil() as usize,
        pfa: 1e-6,
        minimum_snr_db: 12.,
    };
    cfar.threshold(fft_len)?;
    let wave = &z[tx0..tx1];
    let energy = wave.iter().map(|z| z.norm_sqr() as f64).sum::<f64>();
    let mut delay_guard = 25usize;
    for lag in (25..wave.len()).step_by(25) {
        let cross = hard_target_signal::single::cross_products(
            &wave[..wave.len() - lag],
            &wave[lag..],
            wave.len() - lag,
        )
        .unwrap()[0];
        if (cross.norm_sqr() as f64) < 0.5 * energy * energy {
            delay_guard = lag;
            break;
        }
    }
    let frequency_guard = fs / (tx1 - tx0) as f32;
    let mut detections = vec![];
    let mut intervals = 0;
    // Keep each coherent interval independent; do not wrap receive data at IPPs.
    for start in (0..z.len().saturating_sub(n + max_delay) + 1).step_by(n) {
        let mut tx = z[start..start + n].to_vec();
        for (i, v) in tx.iter_mut().enumerate() {
            if i % ipp < tx0 || i % ipp >= tx1 {
                *v = C::default();
            }
        }
        let mut rx = z[start..start + n + max_delay].to_vec();
        for (i, v) in rx.iter_mut().enumerate() {
            let t = (start + i) % ipp;
            if (tx0..tx1).contains(&t) || v.re.abs() >= 32767. || v.im.abs() >= 32767. {
                *v = C::default();
            }
        }
        let began = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let mut hits = if let Some(g) = gpu.as_mut() {
            g.detect(&tx, &rx, &s, &cfar)?
        } else {
            detect_cpu(&tx, &rx, &s, &cfar)?
        };
        #[cfg(not(feature = "cuda"))]
        let mut hits = detect_cpu(&tx, &rx, &s, &cfar)?;
        hits.sort_by(|a, b| b.power.total_cmp(&a.power));
        let mut kept: Vec<hard_target::Detection> = vec![];
        // Deduplicate nearby code/acceleration maxima; no truncation by count.

        for hit in hits {
            if !kept.iter().any(|p| {
                p.delay_samples.abs_diff(hit.delay_samples) <= delay_guard
                    && (p.doppler_hz - hit.doppler_hz).abs() <= frequency_guard
            }) {
                kept.push(hit);
            }
        }
        let count = kept.len();
        for p in kept {
            let pulse_sample = b.start_sample + start as u64 + tx0 as u64;
            detections.push(serde_json::json!({"pulse_start_sample":pulse_sample,"pulse_time_unix_s":pulse_sample as f64/source.sample_rate_hz as f64,"pulse_count":pulses,"coherent_duration_s":n as f32/fs,"range_m":p.delay_samples as f32/fs*299792458./2.,"doppler_hz":p.doppler_hz,"doppler_rate_hz_s":p.doppler_rate_hz_s,"snr_db":p.snr_db}));
        }
        intervals += 1;
        println!(
            "interval={intervals} backend={} elapsed={:?} cfar_detections={count}",
            args[3],
            began.elapsed()
        );
    }
    let h5 = hdf5::File::create(dir.join("detections.h5"))?;
    let times: Vec<f64> = detections
        .iter()
        .map(|d| d["pulse_time_unix_s"].as_f64().unwrap())
        .collect();
    let samples: Vec<u64> = detections
        .iter()
        .map(|d| d["pulse_start_sample"].as_u64().unwrap())
        .collect();
    h5.new_dataset_builder()
        .with_data(&times)
        .create("pulse_time_unix_s")?;
    h5.new_dataset_builder()
        .with_data(&samples)
        .create("pulse_start_sample")?;
    for key in ["range_m", "doppler_hz", "snr_db", "doppler_rate_hz_s"] {
        let v: Vec<f32> = detections
            .iter()
            .map(|d| d[key].as_f64().unwrap() as f32)
            .collect();
        h5.new_dataset_builder().with_data(&v).create(key)?;
    }
    let manifest = serde_json::json!({"schema":"hard-target-cfar-v1","mode":args[4],"backend":args[3],"source_file":Path::new(&args[1]).file_name().unwrap().to_string_lossy(),"start_unix_s":b.start_sample as f64/source.sample_rate_hz as f64,"end_unix_s":(b.start_sample+z.len()as u64)as f64/source.sample_rate_hz as f64,"sample_rate_hz":fs,"tx_window_samples":[tx0,tx1],"cfar":cfar,"non_maximum_suppression_delay_samples":delay_guard,"non_maximum_suppression_frequency_hz":frequency_guard,"intervals":intervals,"detections":detections,"precision":"float32/complex64; float64 UTC seconds and uint64 sample times","status":"CFAR detections; object classification unconfirmed","limitations":["Nominal per-cell PFA assumes independent exponential training cells; waveform-induced correlation and the full search trial count are not included","No full matched-filter maps are retained","Coarse fixed-delay search; range migration not refined","Range is slant range; positive Doppler follows recorded IQ, not an established physical velocity sign"]});
    let metadata = serde_json::to_vec(&manifest)?;
    h5.new_dataset_builder()
        .with_data(&metadata)
        .create("metadata_json_utf8")?;
    h5.flush()?;
    std::fs::write(dir.join("manifest.json"), metadata)?;
    Ok(())
}
