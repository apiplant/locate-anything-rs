//! `LocateAnythingForConditionalGeneration` + its `generate` loop
//! (modeling_locateanything.py).

use crate::config::{Config, PreprocessorConfig, TokenIds};
use crate::image_proc::{self, ProcessedImage};
use crate::qwen2::{MaskKind, Qwen2};
use crate::sampling::{decode_block, handle_pattern, sample_tokens, BlockType, GenerationMode, SamplingParams};
use crate::tokenizer;
use crate::vision::{MoonVit, Projector};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const MODEL_MAX_LENGTH: usize = 16384;

#[derive(Debug, Clone, Copy)]
enum Next {
    Mtp,
    Ar,
    Stop,
}

pub struct GenerateOptions {
    pub mode: GenerationMode,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub seed: u64,
    pub verbose: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerateStats {
    pub num_tokens: usize,
    pub forward_steps: usize,
    pub switch_to_ar: usize,
    pub prefill_secs: f64,
    pub vision_secs: f64,
    /// Decode-phase (after prefill) forward passes incl. logits transfer.
    pub forward_secs: f64,
    /// Host-side logits processing / sampling / block decoding.
    pub sampling_secs: f64,
    pub total_secs: f64,
}

pub struct LocateAnything {
    pub config: Config,
    pub preprocessor: PreprocessorConfig,
    pub tokenizer: Tokenizer,
    vision: MoonVit,
    projector: Projector,
    lm: Qwen2,
    token_ids: TokenIds,
    device: Device,
}

impl LocateAnything {
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let config = Config::load(model_dir)?;
        let preprocessor = PreprocessorConfig::load(model_dir)?;
        let tokenizer = tokenizer::load(model_dir)?;

        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("model.safetensors.index.json"))
                .context("reading model.safetensors.index.json")?,
        )?;
        let mut files: Vec<String> = index["weight_map"]
            .as_object()
            .context("weight_map missing")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        let paths: Vec<_> = files.iter().map(|f| model_dir.join(f)).collect();
        for p in &paths {
            let len = std::fs::metadata(p).with_context(|| format!("missing {}", p.display()))?.len();
            if len < 1024 {
                bail!("{} looks like a git-lfs pointer; run `git lfs pull`", p.display());
            }
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = device {
            // Single-stream use: cudarc's cross-stream event tracking is pure
            // overhead (an event create/destroy pair per allocation).
            unsafe { d.disable_event_tracking() };
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? };

        let vc = &config.vision_config;
        let merged = vc.hidden_size * vc.merge_kernel_size[0] * vc.merge_kernel_size[1];
        let vision = MoonVit::new(vc, vb.pp("vision_model"))?;
        let projector = Projector::new(merged, config.text_config.hidden_size, vb.pp("mlp1"))?;
        let lm = Qwen2::new(&config.text_config, vb.pp("language_model"))?;
        let token_ids = config.token_ids();
        Ok(Self { config, preprocessor, tokenizer, vision, projector, lm, token_ids, device: device.clone() })
    }

    pub fn preprocess(&self, img: &image::RgbImage, in_token_limit: Option<usize>) -> Result<ProcessedImage> {
        image_proc::preprocess(img, &self.preprocessor, in_token_limit.unwrap_or(self.preprocessor.in_token_limit))
    }

    /// Vision tower + projector: `[num_image_tokens, hidden]`.
    pub fn encode_image(&self, img: &ProcessedImage) -> Result<Tensor> {
        let feats = self.vision.forward(&img.patches, img.grid_hw)?;
        Ok(self.projector.forward(&feats)?)
    }

    /// Builds the prompt exactly as `py_apply_chat_template` + `replace_media_placeholder`.
    pub fn build_prompt(&self, question: &str, num_image_tokens: usize) -> Result<Vec<u32>> {
        let tok = &self.tokenizer;
        let mut ids = tokenizer::encode(
            tok,
            "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n<image 1>",
        )?;
        ids.push(tokenizer::token_id(tok, "<img>")?);
        ids.extend(std::iter::repeat(self.config.image_token_index).take(num_image_tokens));
        ids.push(tokenizer::token_id(tok, "</img>")?);
        ids.extend(tokenizer::encode(tok, &format!("{question}<|im_end|>\n<|im_start|>assistant\n"))?);
        Ok(ids)
    }

    pub fn generate(&mut self, img: &ProcessedImage, question: &str, opts: &GenerateOptions) -> Result<(String, GenerateStats)> {
        let t = self.token_ids;
        let block = self.config.text_config.block_size;
        let vocab = self.config.text_config.vocab_size;
        let start = Instant::now();

        let vis = self.encode_image(img)?;
        self.device.synchronize()?;
        let vision_secs = start.elapsed().as_secs_f64();

        let mut generated = self.build_prompt(question, img.num_tokens)?;
        let seq_len = generated.len();
        let total_len = MODEL_MAX_LENGTH.min(seq_len + opts.max_new_tokens);
        let img_start = generated
            .iter()
            .position(|&x| x == self.config.image_token_index)
            .context("prompt has no image tokens")?;

        // Unique token ids seen so far (prompt + output), for the repetition penalty.
        let mut is_seen = vec![false; vocab];
        let mut seen: Vec<u32> = Vec::new();
        let mut mark_seen = |toks: &[u32], seen: &mut Vec<u32>| {
            for &x in toks {
                if let Some(s) = is_seen.get_mut(x as usize) {
                    if !*s {
                        *s = true;
                        seen.push(x);
                    }
                }
            }
        };
        mark_seen(&generated, &mut seen);

        let mut rng = StdRng::seed_from_u64(opts.seed);
        let mut use_mtp = opts.mode != GenerationMode::Slow;
        let (mut steps, mut switch_to_ar) = (0usize, 0usize);
        let mut prefill_secs = None;
        let (mut forward_secs, mut sampling_secs) = (0f64, 0f64);
        self.lm.clear_cache();

        while generated.len() < total_len {
            steps += 1;
            let len = generated.len();
            let past = self.lm.cache_len();

            let mut ids: Vec<u32> = generated[past..].to_vec();
            let mut positions: Vec<u32> = (past as u32..len as u32).collect();
            let mask = if use_mtp {
                // Duplicate the last token and append block-1 mask tokens; the
                // window's positions restart at the last token's position.
                ids.push(generated[len - 1]);
                ids.extend(std::iter::repeat(t.mask).take(block - 1));
                positions.extend((len - 1..len - 1 + block).map(|p| p as u32));
                MaskKind::Mtp { block }
            } else {
                MaskKind::Causal
            };

            let mut embeds = self.lm.embed(&ids)?;
            if past == 0 {
                let n = img.num_tokens;
                let s = embeds.dim(0)?;
                embeds = Tensor::cat(
                    &[
                        embeds.narrow(0, 0, img_start)?,
                        vis.to_dtype(embeds.dtype())?,
                        embeds.narrow(0, img_start + n, s - img_start - n)?,
                    ],
                    0,
                )?;
            }

            let n_logits = if use_mtp { block } else { 1 };
            let t_fwd = Instant::now();
            let logits = self.lm.forward(&embeds, &positions, mask, n_logits)?;
            // Drop the KV entries of the MTP window; only committed tokens stay cached.
            self.lm.truncate_cache(len)?;
            let rows: Vec<Vec<f32>> = logits.to_vec2()?;
            if past > 0 {
                forward_secs += t_fwd.elapsed().as_secs_f64();
            }
            let t_samp = Instant::now();

            let (probs, x0) = sample_tokens(rows, &seen, &opts.sampling, &mut rng);
            let (next, out, label) = if use_mtp {
                let block_toks = decode_block(&probs, &x0, &t, opts.mode);
                let (kind, out) = handle_pattern(&block_toks, &t, opts.mode);
                let next = match kind {
                    BlockType::ImEnd => Next::Stop,
                    BlockType::ErrorBox => Next::Ar,
                    _ => Next::Mtp,
                };
                (next, out, format!("{kind:?}"))
            } else {
                let tok = x0[0];
                let next = match opts.mode {
                    // Hybrid AR only completes the current box, then returns to MTP.
                    GenerationMode::Hybrid if tok == t.box_end => Next::Mtp,
                    GenerationMode::Hybrid if t.is_coord(tok) || tok == t.none => Next::Ar,
                    GenerationMode::Hybrid => Next::Stop,
                    _ if tok == t.im_end => Next::Stop,
                    _ => Next::Ar,
                };
                (next, vec![tok], format!("{next:?}"))
            };

            sampling_secs += t_samp.elapsed().as_secs_f64();
            if opts.verbose {
                let tag = if use_mtp { "mtp" } else { "ar" };
                eprintln!("[{steps:>4}] {tag:<3} {label:<10} {}", tokenizer::decode(&self.tokenizer, &out)?);
            }
            mark_seen(&out, &mut seen);
            generated.extend_from_slice(&out);

            match next {
                Next::Stop => break,
                // Fast mode never leaves MTP and slow mode never enters it.
                Next::Ar if use_mtp && opts.mode == GenerationMode::Hybrid => {
                    use_mtp = false;
                    switch_to_ar += 1;
                }
                Next::Mtp if !use_mtp && opts.mode == GenerationMode::Hybrid => use_mtp = true,
                _ => {}
            }
            prefill_secs.get_or_insert_with(|| start.elapsed().as_secs_f64());
        }

        let new = &generated[seq_len..];
        let answer = tokenizer::decode(&self.tokenizer, new)?;
        let stats = GenerateStats {
            num_tokens: new.len(),
            forward_steps: steps,
            switch_to_ar,
            prefill_secs: prefill_secs.unwrap_or_else(|| start.elapsed().as_secs_f64()),
            vision_secs,
            forward_secs,
            sampling_secs,
            total_secs: start.elapsed().as_secs_f64(),
        };
        Ok((answer, stats))
    }
}
