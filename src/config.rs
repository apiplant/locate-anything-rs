use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub eos_token_id: u32,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub causal_attn: bool,
    #[serde(default = "default_null")]
    pub null_token_id: u32,
    #[serde(default = "default_switch")]
    pub switch_token_id: u32,
    #[serde(default = "default_text_mask")]
    pub text_mask_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub init_pos_emb_height: usize,
    pub init_pos_emb_width: usize,
    pub merge_kernel_size: [usize; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_index: u32,
    pub box_start_token_id: u32,
    pub box_end_token_id: u32,
    pub coord_start_token_id: u32,
    pub coord_end_token_id: u32,
    pub ref_start_token_id: u32,
    pub ref_end_token_id: u32,
    pub none_token_id: u32,
}

fn default_block_size() -> usize {
    6
}
fn default_null() -> u32 {
    152678
}
fn default_switch() -> u32 {
    152679
}
fn default_text_mask() -> u32 {
    151676
}

impl Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let p = model_dir.join("config.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        Ok(serde_json::from_str(&s)?)
    }

    pub fn token_ids(&self) -> TokenIds {
        TokenIds {
            box_start: self.box_start_token_id,
            box_end: self.box_end_token_id,
            coord_start: self.coord_start_token_id,
            coord_end: self.coord_end_token_id,
            ref_start: self.ref_start_token_id,
            ref_end: self.ref_end_token_id,
            none: self.none_token_id,
            null: self.text_config.null_token_id,
            im_end: self.text_config.eos_token_id,
            switch: self.text_config.switch_token_id,
            mask: self.text_config.text_mask_token_id,
        }
    }
}

/// Mirror of `get_token_ids_from_config` in generate_utils.py.
#[derive(Debug, Clone, Copy)]
pub struct TokenIds {
    pub box_start: u32,
    pub box_end: u32,
    pub coord_start: u32,
    pub coord_end: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub none: u32,
    pub null: u32,
    pub im_end: u32,
    pub switch: u32,
    pub mask: u32,
}

impl TokenIds {
    pub fn is_coord(&self, t: u32) -> bool {
        (self.coord_start..=self.coord_end).contains(&t)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub in_token_limit: usize,
    pub merge_kernel_size: [usize; 2],
    pub patch_size: usize,
}

impl PreprocessorConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let p = model_dir.join("preprocessor_config.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        Ok(serde_json::from_str(&s)?)
    }
}
