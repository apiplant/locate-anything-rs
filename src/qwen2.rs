//! Qwen2 language model with the block-diffusion attention masks used by
//! LocateAnything's Parallel Box Decoding (modeling_qwen2.py + mask_magi_utils.py).

use crate::attention::{chunked_attention, flash, flash_usable};
use crate::config::TextConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

/// Attention pattern for one forward call. The queries are the last `q_len`
/// positions of a `kv_len`-long sequence (`past = kv_len - q_len` are cached).
#[derive(Debug, Clone, Copy)]
pub enum MaskKind {
    /// Plain causal attention (prefill / auto-regressive decoding).
    Causal,
    /// MTP step: the last `block` queries form a bidirectional window that
    /// sees the prefix except the key right before the window (the real copy
    /// of the duplicated last token); earlier queries are causal.
    Mtp { block: usize },
}

impl MaskKind {
    /// Additive mask `[q_len, kv_len]` (0 / -inf) for queries at the end of the sequence.
    fn build(self, q_len: usize, kv_len: usize) -> Vec<f32> {
        let past = kv_len - q_len;
        let mut m = vec![0f32; q_len * kv_len];
        for (qi, row) in m.chunks_mut(kv_len).enumerate() {
            match self {
                MaskKind::Mtp { block } if qi >= q_len - block => row[kv_len - block - 1] = f32::NEG_INFINITY,
                _ => row[past + qi + 1..].fill(f32::NEG_INFINITY),
            }
        }
        m
    }
}

/// Pre-allocated per-layer KV cache `[kv_heads, capacity, head_dim]`; appends
/// write in place and truncation just moves the length.
struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl KvCache {
    const fn new() -> Self {
        Self { k: None, v: None, len: 0 }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (nkv, s, hd) = k.dims3()?;
        let need = self.len + s;
        let cap = self.k.as_ref().map_or(0, |t| t.dim(1).unwrap_or(0));
        if need > cap {
            let new_cap = need.max(2 * cap).max(1024).next_multiple_of(256);
            let grow = |old: &Option<Tensor>| -> Result<Tensor> {
                let t = Tensor::zeros((nkv, new_cap, hd), k.dtype(), k.device())?;
                if let Some(old) = old {
                    t.slice_set(&old.narrow(1, 0, self.len)?.contiguous()?, 1, 0)?;
                }
                Ok(t)
            };
            self.k = Some(grow(&self.k)?);
            self.v = Some(grow(&self.v)?);
        }
        let (kc, vc) = (self.k.as_ref().unwrap(), self.v.as_ref().unwrap());
        kc.slice_set(k, 1, self.len)?;
        vc.slice_set(v, 1, self.len)?;
        self.len = need;
        Ok((kc.narrow(1, 0, need)?, vc.narrow(1, 0, need)?))
    }
}

/// Linear layer whose bias is added from cached contiguous `[S, N]` tiles:
/// broadcast adds are strided ops, which cost candle a host->device upload.
struct TiledLinear {
    w_t: Tensor,
    bias: Tensor,
    tiles: std::collections::HashMap<usize, Tensor>,
}

impl TiledLinear {
    fn new(w: Tensor, bias: Tensor) -> Result<Self> {
        Ok(Self { w_t: w.t()?, bias, tiles: Default::default() })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let s = x.dim(0)?;
        let y = x.matmul(&self.w_t)?;
        if s > 64 {
            return y.broadcast_add(&self.bias);
        }
        let tile = match self.tiles.get(&s) {
            Some(t) => t,
            None => {
                let t = self.bias.unsqueeze(0)?.repeat((s, 1))?;
                self.tiles.entry(s).or_insert(t)
            }
        };
        y + tile
    }
}

struct Attention {
    qkv: TiledLinear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: KvCache,
}

impl Attention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = h / cfg.num_attention_heads;
        let kv = cfg.num_key_value_heads * hd;
        // q/k/v fused into one matmul.
        let w = Tensor::cat(
            &[vb.get((h, h), "q_proj.weight")?, vb.get((kv, h), "k_proj.weight")?, vb.get((kv, h), "v_proj.weight")?],
            0,
        )?;
        let b = Tensor::cat(&[vb.get(h, "q_proj.bias")?, vb.get(kv, "k_proj.bias")?, vb.get(kv, "v_proj.bias")?], 0)?;
        Ok(Self {
            qkv: TiledLinear::new(w, b)?,
            o_proj: linear_no_bias(h, h, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            cache: KvCache::new(),
        })
    }

    /// `window`: size of the trailing MTP window among the queries (0 for causal).
    fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, window: usize) -> Result<Tensor> {
        let (s, h) = x.dims2()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let qkv = self.qkv.forward(x)?;
        // [S, n*D] -> [1, n, S, D] for rope
        let heads = |off: usize, n: usize| {
            qkv.narrow(1, off * hd, n * hd)?.reshape((s, n, hd))?.transpose(0, 1)?.unsqueeze(0)?.contiguous()
        };
        let q = candle_nn::rotary_emb::rope(&heads(0, nh)?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&heads(nh, nkv)?, cos, sin)?.squeeze(0)?;
        let v = heads(nh + nkv, nkv)?.squeeze(0)?;

        let (k, v) = self.cache.append(&k, &v)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let out = if s > 64 && flash_usable(x) {
            // Prefill: causal rows through FlashAttention; only the trailing
            // MTP window (if any) needs the custom mask.
            let win = window;
            let nc = s - win;
            let kv_c = k.dim(1)? - win;
            let bthd = |t: Tensor| t.unsqueeze(0)?.transpose(1, 2)?.contiguous();
            let qc = q.squeeze(0)?.narrow(1, 0, nc)?;
            let causal = flash(
                &bthd(qc)?,
                &bthd(k.narrow(1, 0, kv_c)?)?,
                &bthd(v.narrow(1, 0, kv_c)?)?,
                scale,
                true,
            )?
            .squeeze(0)?
            .reshape((nc, h))?;
            if win == 0 {
                causal
            } else {
                let qw = q.narrow(2, nc, win)?.reshape((nkv, nh / nkv, win, hd))?;
                let mw = mask.map(|m| m.narrow(0, nc, win)).transpose()?;
                let w = chunked_attention(&qw, &k, &v, scale, mw.as_ref())?;
                let w = w.reshape((nh, win, hd))?.transpose(0, 1)?.reshape((win, h))?;
                Tensor::cat(&[causal, w], 0)?
            }
        } else {
            // GQA without repeat_kv: head h uses kv head h / (nh / nkv).
            let q = q.reshape((nkv, nh / nkv, s, hd))?;
            let out = chunked_attention(&q, &k, &v, scale, mask)?;
            out.reshape((nh, s, hd))?.transpose(0, 1)?.reshape((s, h))?
        };
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&(self.gate.forward(x)?.silu()? * self.up.forward(x)?)?)
    }
}

struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            ln1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, window: usize) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?, cos, sin, mask, window)?)?;
        &x + self.mlp.forward(&self.ln2.forward(&x)?)?
    }
}

pub struct Qwen2 {
    embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    inv_freq: Vec<f32>,
    num_heads: usize,
    device: Device,
    dtype: DType,
}

impl Qwen2 {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let m = vb.pp("model");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(cfg, m.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(m.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?, None)
        };
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        let inv_freq = (0..hd / 2)
            .map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / hd as f32))
            .collect();
        Ok(Self {
            embed: embedding(cfg.vocab_size, cfg.hidden_size, m.pp("embed_tokens"))?,
            layers,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, m.pp("norm"))?,
            lm_head,
            inv_freq,
            num_heads: cfg.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, &self.device)?;
        self.embed.forward(&ids)
    }

    pub fn cache_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.attn.cache.len)
    }

    pub fn truncate_cache(&mut self, len: usize) -> Result<()> {
        for l in &mut self.layers {
            l.attn.cache.len = l.attn.cache.len.min(len);
        }
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        for l in &mut self.layers {
            l.attn.cache.len = 0;
        }
    }

    fn rope_tables(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for f in &self.inv_freq {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let shape = (positions.len(), half);
        Ok((
            Tensor::from_vec(cos, shape, &self.device)?.to_dtype(self.dtype)?,
            Tensor::from_vec(sin, shape, &self.device)?.to_dtype(self.dtype)?,
        ))
    }

    /// Runs the decoder over `embeds` `[S, H]` at the given absolute positions
    /// (appending to the KV cache) and returns f32 logits for the last
    /// `n_logits` positions, `[n_logits, vocab]`.
    pub fn forward(&mut self, embeds: &Tensor, positions: &[u32], mask: MaskKind, n_logits: usize) -> Result<Tensor> {
        let (cos, sin) = self.rope_tables(positions)?;
        let s = positions.len();
        let kv_len = self.cache_len() + s;
        let window = match mask {
            MaskKind::Mtp { block } => block,
            MaskKind::Causal => 0,
        };
        let mask = match mask {
            MaskKind::Causal if s == 1 => None,
            _ => {
                let m = Tensor::from_vec(mask.build(s, kv_len), (s, kv_len), &self.device)?.to_dtype(self.dtype)?;
                // Decode steps: expand to the full score shape once so every
                // layer does a contiguous add instead of a strided broadcast.
                let nh = self.num_heads;
                Some(if s <= 64 { m.unsqueeze(0)?.repeat((nh, 1, 1))? } else { m })
            }
        };
        let mut x = embeds.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, &cos, &sin, mask.as_ref(), window)?;
        }
        let s = x.dim(0)?;
        let x = self.norm.forward(&x.narrow(0, s - n_logits, n_logits)?)?;
        self.lm_head.forward(&x)?.to_dtype(DType::F32)
    }
}

