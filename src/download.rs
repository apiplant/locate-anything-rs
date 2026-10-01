//! Downloads the LocateAnything-3B checkpoint from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/locate-anything-rs/LocateAnything-3B`, falling back to
//! `~/.cache/locate-anything-rs/LocateAnything-3B` when `$XDG_CACHE_HOME` is unset. Files are
//! fetched to a `.part` sibling and renamed into place once complete, so a killed download never
//! leaves a checkpoint that looks done.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// The Hugging Face repository the weights come from.
pub const HF_REPO: &str = "nvidia/LocateAnything-3B";

/// Files besides the weight shards that a checkpoint needs in order to load.
const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
];

/// `$XDG_CACHE_HOME/locate-anything-rs`, defaulting to `~/.cache/locate-anything-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("locate-anything-rs"))
}

/// Where [`download`] puts (or would put) the checkpoint.
pub fn checkpoint_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(HF_REPO.rsplit('/').next().unwrap_or(HF_REPO)))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension(match dest.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut written: u64 = 0;
    let mut last_report = 0u64;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        // The weight shards are gigabytes: show progress about every 64 MiB.
        if written - last_report >= 64 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!("\r  {name} done ({} MiB)          ", written >> 20);
    Ok(())
}

/// The weight shard file names listed in `model.safetensors.index.json`.
fn shard_names(index: &Path) -> Result<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index).context("reading model.safetensors.index.json")?)?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).context("index has no weight_map")?;
    let mut names: Vec<String> = map.values().filter_map(|s| s.as_str().map(str::to_owned)).collect();
    names.sort();
    names.dedup();
    Ok(names)
}

/// Downloads the checkpoint into the cache directory, skipping files already present, and returns
/// the checkpoint directory. Needs network access on first use; any failure (offline, 404, disk
/// error, ...) is returned as an error and nothing partial is left looking complete.
pub fn download() -> Result<PathBuf> {
    let dir = checkpoint_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let url = |file: &str| format!("https://huggingface.co/{HF_REPO}/resolve/main/{file}");
    let mut announced = false;
    let mut fetch = |file: &str| -> Result<()> {
        let dest = dir.join(file);
        if dest.is_file() {
            return Ok(());
        }
        if !announced {
            eprintln!("Downloading {HF_REPO} (about 7.6 GB) from https://huggingface.co/{HF_REPO} into {}", dir.display());
            announced = true;
        }
        fetch_to_file(&url(file), &dest)
    };

    for file in REQUIRED_FILES {
        fetch(file)?;
    }
    for shard in shard_names(&dir.join("model.safetensors.index.json"))? {
        fetch(&shard)?;
    }
    Ok(dir)
}
