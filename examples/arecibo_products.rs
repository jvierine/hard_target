use anyhow::{ensure, Result};
use hard_target::arecibo::{ion_lpi, plasma, Recording};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 5, "input.h5 output-directory cpu|gpu ion|plasma");
    let r = Recording::open(Path::new(&a[1]))?;
    match a[4].as_str() {
        "plasma" => plasma(&r, Path::new(&a[2]), a[3] == "gpu"),
        "ion" => {
            let root = Path::new(&a[2]).parent().unwrap();
            let mut hits = vec![];
            for mode in ["space", "meteor"] {
                let m: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(root.join(mode).join("manifest.json"))?)?;
                if let Some(v) = m["detections"].as_array() {
                    hits.extend(v.clone());
                }
            }
            ion_lpi(&r, Path::new(&a[2]), &hits)
        }
        _ => anyhow::bail!("unknown product"),
    }
}
