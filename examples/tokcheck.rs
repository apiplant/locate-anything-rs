//! Prints token ids for each line of stdin (for comparison with the HF tokenizer).
use std::io::BufRead;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or("/mnt/extra/ai/LocateAnything-3B".into());
    let tok = locate_anything::tokenizer::load(std::path::Path::new(&dir))?;
    for line in std::io::stdin().lock().lines() {
        let line = line?.replace("\\n", "\n");
        let ids = locate_anything::tokenizer::encode(&tok, &line)?;
        println!("{ids:?}");
        println!("{:?}", locate_anything::tokenizer::decode(&tok, &ids)?);
    }
    Ok(())
}
