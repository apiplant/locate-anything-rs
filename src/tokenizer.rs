//! Qwen2 byte-level BPE tokenizer built from `vocab.json` + `merges.txt` +
//! `tokenizer_config.json` (the HF repo ships no `tokenizer.json`).

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};
use std::path::Path;
use tokenizers::Tokenizer;

const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub fn load(model_dir: &Path) -> Result<Tokenizer> {
    let tj = model_dir.join("tokenizer.json");
    if tj.exists() {
        return Tokenizer::from_file(&tj).map_err(|e| anyhow!("loading {}: {e}", tj.display()));
    }

    let vocab: Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("vocab.json")).context("reading vocab.json")?,
    )?;
    let merges: Vec<Value> = std::fs::read_to_string(model_dir.join("merges.txt"))
        .context("reading merges.txt")?
        .lines()
        .filter(|l| !l.starts_with("#version") && !l.is_empty())
        .map(|l| Value::String(l.to_string()))
        .collect();

    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .context("reading tokenizer_config.json")?,
    )?;
    let decoder = cfg["added_tokens_decoder"]
        .as_object()
        .ok_or_else(|| anyhow!("tokenizer_config.json has no added_tokens_decoder"))?;
    let mut added: Vec<(u64, Map<String, Value>)> = decoder
        .iter()
        .map(|(id, v)| {
            let mut m = v.as_object().cloned().unwrap_or_default();
            let id: u64 = id.parse().unwrap_or(0);
            m.insert("id".into(), json!(id));
            (id, m)
        })
        .collect();
    added.sort_by_key(|(id, _)| *id);
    let added: Vec<Value> = added.into_iter().map(|(_, m)| Value::Object(m)).collect();

    let byte_level = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false
    });
    let spec = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": { "type": "NFC" },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                { "type": "Split", "pattern": { "Regex": QWEN2_PRETOKENIZE_REGEX },
                  "behavior": "Isolated", "invert": false },
                byte_level,
            ]
        },
        "post_processor": null,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        }
    });
    Tokenizer::from_bytes(serde_json::to_vec(&spec)?).map_err(|e| anyhow!("building tokenizer: {e}"))
}

pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec())
}

pub fn decode(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    tok.decode(ids, false).map_err(|e| anyhow!("detokenize: {e}"))
}

pub fn token_id(tok: &Tokenizer, s: &str) -> Result<u32> {
    tok.token_to_id(s).ok_or_else(|| anyhow!("token {s:?} missing from vocab"))
}
