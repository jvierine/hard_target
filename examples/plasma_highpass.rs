use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 2, "plasma-directory");
    hard_target::arecibo::plasma_highpass(std::path::Path::new(&a[1]))
}
