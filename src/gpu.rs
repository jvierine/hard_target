//! Bounded GPU tiles: upload IQ once, match/phase/FFT on device, reuse plans.
use super::*;
use cudarc::{
    cufft::{
        sys::{cufftType, float2},
        CudaFft, FftDirection,
    },
    driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::{collections::HashMap, sync::Arc};
pub struct GpuBank {
    context: Arc<CudaContext>,
    matched: CudaFunction,
    pointwise: CudaFunction,
    rotate: CudaFunction,
    cfar: CudaFunction,
    plans: HashMap<(usize, usize), CudaFft>,
    pub tile_ranges: usize,
    pub memory_budget_bytes: usize,
}
fn device(z: &[C]) -> Vec<float2> {
    z.iter().map(|z| float2 { x: z.re, y: z.im }).collect()
}
impl GpuBank {
    pub fn new(device: usize) -> Result<Self> {
        let context = CudaContext::new(device)?;
        let module = context.load_module(Ptx::from_src(include_str!("bank.ptx")))?;
        Ok(Self {
            pointwise: module.load_function("match_pointwise")?,
            cfar: module.load_function("cfar_detect")?,
            matched: module.load_function("match_bins")?,
            rotate: module.load_function("rotate_bins")?,
            context,
            plans: HashMap::new(),
            tile_ranges: 64,
            memory_budget_bytes: 512 * 1024 * 1024,
        })
    }
    pub fn search(&mut self, tx: &[C], rx: &[C], s: &Settings) -> Result<Output> {
        self.search_mode(tx, rx, s, false)
    }
    pub fn mean_power(&mut self, tx: &[C], rx: &[C], s: &Settings) -> Result<Output> {
        ensure!(
            s.doppler_rates_hz_s.len() == 1,
            "mean power requires one rate"
        );
        self.search_mode(tx, rx, s, true)
    }
    fn search_mode(
        &mut self,
        tx: &[C],
        rx: &[C],
        s: &Settings,
        mean_power: bool,
    ) -> Result<Output> {
        let n = validate(tx, rx, s)?;
        ensure!(self.tile_ranges > 0, "zero GPU tile size");
        let budget = tx
            .len()
            .checked_add(rx.len())
            .and_then(|v| {
                v.checked_add(n.checked_mul(self.tile_ranges.checked_mul(6)?.checked_add(2)?)?)
            })
            .and_then(|v| v.checked_mul(8));
        ensure!(
            budget.is_some_and(|v| v <= self.memory_budget_bytes),
            "GPU memory budget exceeded; shorten coherent interval or tile"
        );
        let stream = self.context.default_stream();
        let dtx = stream.clone_htod(&device(tx))?;
        let drx = stream.clone_htod(&device(rx))?;
        let mut peaks = vec![];
        let mut power = vec![];
        for ranges in s.delays.chunks(self.tile_ranges) {
            let count = n * ranges.len();
            let threads = u32::try_from(
                count
                    .checked_mul(if s.decimation == 1 { 1 } else { 32 })
                    .ok_or_else(|| anyhow::anyhow!("thread overflow"))?,
            )?;
            let mut echo = stream.alloc_zeros::<float2>(count)?;
            let mut rotated = stream.alloc_zeros::<float2>(count)?;
            let mut transformed = stream.alloc_zeros::<float2>(count)?;
            let offsets: Vec<u64> = ranges.iter().map(|&r| r as u64).collect();
            let offsets = stream.clone_htod(&offsets)?;
            let (d, nn, total) = (s.decimation as u64, n as u64, count as u64);
            let mut launch = stream.launch_builder(if s.decimation == 1 {
                &self.pointwise
            } else {
                &self.matched
            });
            launch
                .arg(&dtx)
                .arg(&drx)
                .arg(&mut echo)
                .arg(&d)
                .arg(&nn)
                .arg(&offsets)
                .arg(&total);
            unsafe {
                launch.launch(LaunchConfig::for_num_elems(threads))?;
            }
            if !self.plans.contains_key(&(n, ranges.len())) {
                self.plans.insert(
                    (n, ranges.len()),
                    CudaFft::plan_1d(
                        i32::try_from(n)?,
                        cufftType::CUFFT_C2C,
                        i32::try_from(ranges.len())?,
                        stream.clone(),
                    )?,
                );
            }
            let plan = &self.plans[&(n, ranges.len())];
            let mut best: Vec<_> = ranges.iter().map(|&r| empty_peak(r)).collect();
            let mut display = vec![0.; ranges.len() * s.display_bins];
            for &rate in &s.doppler_rates_hz_s {
                let ph = stream.clone_htod(&device(&phase(s, n, rate)))?;
                let mut launch = stream.launch_builder(&self.rotate);
                launch
                    .arg(&echo)
                    .arg(&ph)
                    .arg(&mut rotated)
                    .arg(&nn)
                    .arg(&total);
                unsafe {
                    launch.launch(LaunchConfig::for_num_elems(u32::try_from(count)?))?;
                }
                plan.exec_c2c(&mut rotated, &mut transformed, FftDirection::Forward)?;
                let host = stream.clone_dtoh(&transformed)?;
                for (j, row) in host.chunks_exact(n).enumerate() {
                    let z: Vec<C> = row.iter().map(|v| C::new(v.x, v.y)).collect();
                    summarize(
                        s,
                        n,
                        ranges[j],
                        rate,
                        &z,
                        &mut best[j],
                        &mut display[j * s.display_bins..(j + 1) * s.display_bins],
                        mean_power,
                    );
                }
            }
            peaks.extend(best);
            power.extend(display);
        }
        Ok(Output {
            peaks,
            power,
            fft_len: n,
            display_bins: s.display_bins,
        })
    }
}
impl GpuBank {
    /// Detect on the GPU; download only compact CFAR hits, never full FFT maps.
    pub fn detect(
        &mut self,
        tx: &[C],
        rx: &[C],
        s: &Settings,
        c: &CfarSettings,
    ) -> Result<Vec<Detection>> {
        let n = validate(tx, rx, s)?;
        let threshold = c.threshold(n)?;
        ensure!(self.tile_ranges > 0, "zero tile size");
        let estimate = (tx.len() + rx.len() + n * s.delays.len() * 4) * 8;
        ensure!(
            estimate <= self.memory_budget_bytes,
            "GPU memory budget exceeded"
        );
        let stream = self.context.default_stream();
        let dtx = stream.clone_htod(&device(tx))?;
        let drx = stream.clone_htod(&device(rx))?;
        let mut detections = vec![];
        for ranges in s.delays.chunks(s.delays.len()) {
            let count = n * ranges.len();
            let mut echo = stream.alloc_zeros::<float2>(count)?;
            let mut rotated = stream.alloc_zeros::<float2>(count)?;
            let mut transformed = stream.alloc_zeros::<float2>(count)?;
            let offsets =
                stream.clone_htod(&ranges.iter().map(|&r| r as u64).collect::<Vec<_>>())?;
            let (d, nn, total) = (s.decimation as u64, n as u64, count as u64);
            let mut launch = stream.launch_builder(if s.decimation == 1 {
                &self.pointwise
            } else {
                &self.matched
            });
            launch
                .arg(&dtx)
                .arg(&drx)
                .arg(&mut echo)
                .arg(&d)
                .arg(&nn)
                .arg(&offsets)
                .arg(&total);
            unsafe {
                launch.launch(LaunchConfig::for_num_elems(u32::try_from(
                    count * if s.decimation == 1 { 1 } else { 32 },
                )?))?;
            }
            if !self.plans.contains_key(&(n, ranges.len())) {
                self.plans.insert(
                    (n, ranges.len()),
                    CudaFft::plan_1d(
                        i32::try_from(n)?,
                        cufftType::CUFFT_C2C,
                        i32::try_from(ranges.len())?,
                        stream.clone(),
                    )?,
                );
            }
            let plan = &self.plans[&(n, ranges.len())];
            for &rate in &s.doppler_rates_hz_s {
                let ph = stream.clone_htod(&device(&phase(s, n, rate)))?;
                let mut launch = stream.launch_builder(&self.rotate);
                launch
                    .arg(&echo)
                    .arg(&ph)
                    .arg(&mut rotated)
                    .arg(&nn)
                    .arg(&total);
                unsafe {
                    launch.launch(LaunchConfig::for_num_elems(u32::try_from(count)?))?;
                }
                plan.exec_c2c(&mut rotated, &mut transformed, FftDirection::Forward)?;
                let capacity = 16384_u32;
                let mut hits = stream.alloc_zeros::<f32>(capacity as usize * 4)?;
                let mut counter = stream.alloc_zeros::<u32>(1)?;
                let training = u32::try_from(c.training_each_side)?;
                let guard = u32::try_from(c.guard_cells)?;
                let range_train = u32::try_from(c.range_training_each_side)?;
                let range_guard = u32::try_from(c.range_guard_cells)?;
                let row_count = u32::try_from(ranges.len())?;
                let log_pfa = -c.pfa.log2();
                let minimum = 10_f32.powf(c.minimum_snr_db / 10.);
                let mut launch = stream.launch_builder(&self.cfar);
                launch
                    .arg(&transformed)
                    .arg(&mut hits)
                    .arg(&mut counter)
                    .arg(&nn)
                    .arg(&total)
                    .arg(&training)
                    .arg(&guard)
                    .arg(&threshold)
                    .arg(&capacity)
                    .arg(&range_train)
                    .arg(&range_guard)
                    .arg(&row_count)
                    .arg(&log_pfa)
                    .arg(&minimum);
                unsafe {
                    launch.launch(LaunchConfig::for_num_elems(u32::try_from(count)?))?;
                }
                let num = stream.clone_dtoh(&counter)?[0];
                ensure!(
                    num <= capacity,
                    "CFAR hit capacity exceeded; detection output would be incomplete"
                );
                if num > 0 {
                    let compact = stream.clone_dtoh(&hits)?;
                    for v in compact[..num as usize * 4].chunks_exact(4) {
                        let row = v[0] as usize;
                        let index = v[1] as usize;
                        ensure!(row < ranges.len() && index < n, "invalid CFAR device index");
                        detections.push(detection(s, n, ranges[row], rate, index, v[2], v[3]));
                    }
                }
            }
        }
        Ok(detections)
    }
}
