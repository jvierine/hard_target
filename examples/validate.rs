use hard_target::{Settings,search_cpu};
use num_complex::Complex32 as C;
fn main()->anyhow::Result<()>{
 let n=4096;let delay=37;let fd=-77.;let fs=4096.;let rate=24.;
 let tx:Vec<_>=(0..n).map(|i|if i%128<64{C::from_polar(1.,((i*73+i*i*17)%997)as f32)}else{C::default()}).collect();
 let mut rx=vec![C::default();n+80];for i in 0..n{let t=i as f32/fs;rx[i+delay]=tx[i]*C::from_polar(1.,2.*std::f32::consts::PI*fd*t+std::f32::consts::PI*rate*t*t);}
 let s=Settings{sample_rate_hz:fs,decimation:2,delays:(0..80).collect(),doppler_rates_hz_s:vec![0.,24.,48.],display_bins:128};
 let cpu=search_cpu(&tx,&rx,&s)?;let p=cpu.peaks.iter().max_by(|a,b|a.power.total_cmp(&b.power)).unwrap();println!("CPU {p:?}");assert_eq!(p.delay_samples,delay);assert_eq!(p.doppler_hz,fd);assert_eq!(p.doppler_rate_hz_s,rate);
 #[cfg(feature="cuda")]{let mut engine=hard_target::gpu::GpuBank::new(0)?;let gpu=engine.search(&tx,&rx,&s)?;let p=gpu.peaks.iter().max_by(|a,b|a.power.total_cmp(&b.power)).unwrap();println!("GPU {p:?}");assert_eq!(p.delay_samples,delay);assert_eq!(p.doppler_hz,fd);assert_eq!(p.doppler_rate_hz_s,rate);
 for(a,b)in cpu.peaks.iter().zip(gpu.peaks){assert!((a.power-b.power).abs()<=a.power*1e-4+1e-4,"CPU/GPU power mismatch");}
 }
 Ok(())
}
