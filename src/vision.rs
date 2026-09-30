//! MoonViT vision encoder (port of modeling_vit.py) plus the `mlp1` projector
//! from modeling_locateanything.py.

use crate::attention::{chunked_attention, flash, flash_usable};
use crate::config::VisionConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};

const LN_EPS: f64 = 1e-5;
const ROPE_THETA: f64 = 10000.0;

struct EncoderLayer {
    norm0: LayerNorm,
    norm1: LayerNorm,
    wqkv: Linear,
    wo: Linear,
    fc0: Linear,
    fc1: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl EncoderLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let ln = LayerNormConfig { eps: LN_EPS, ..Default::default() };
        Ok(Self {
            norm0: layer_norm(h, ln, vb.pp("norm0"))?,
            norm1: layer_norm(h, ln, vb.pp("norm1"))?,
            wqkv: linear(h, 3 * h, vb.pp("wqkv"))?,
            wo: linear(h, h, vb.pp("wo"))?,
            fc0: linear(h, cfg.intermediate_size, vb.pp("mlp.fc0"))?,
            fc1: linear(cfg.intermediate_size, h, vb.pp("mlp.fc1"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (n, hidden) = x.dims2()?;
        let (nh, hd) = (self.num_heads, self.head_dim);
        let dtype = x.dtype();

        let qkv = self.wqkv.forward(&self.norm0.forward(x)?)?.reshape((n, 3, nh, hd))?;
        // [N, H, D] -> [1, H, N, D]; 2D rope is applied in f32 like the reference.
        let rot = |t: Tensor| -> Result<Tensor> {
            let t = t.to_dtype(DType::F32)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            candle_nn::rotary_emb::rope_i(&t, cos, sin)?.to_dtype(dtype)
        };
        let q = rot(qkv.narrow(1, 0, 1)?.squeeze(1)?)?;
        let k = rot(qkv.narrow(1, 1, 1)?.squeeze(1)?)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        let scale = 1.0 / (hd as f64).sqrt();

        let attn = if flash_usable(x) {
            // [1, H, N, D] -> [1, N, H, D]
            let bnhd = |t: &Tensor| t.transpose(1, 2)?.contiguous();
            flash(&bnhd(&q)?, &bnhd(&k)?, &v.unsqueeze(0)?.contiguous()?, scale, false)?.reshape((n, hidden))?
        } else {
            // Treat every head as its own group: q [H, 1, N, D], k/v [H, N, D].
            let q = q.squeeze(0)?.unsqueeze(1)?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let attn = chunked_attention(&q, &k.squeeze(0)?, &v, scale, None)?;
            attn.squeeze(1)?.transpose(0, 1)?.reshape((n, hidden))?
        };
        let x = (x + self.wo.forward(&attn)?)?;

        let h = self.fc0.forward(&self.norm1.forward(&x)?)?.gelu()?; // tanh approximation
        x + self.fc1.forward(&h)?
    }
}

pub struct MoonVit {
    patch_w: Tensor,
    patch_b: Tensor,
    pos_emb: Vec<f32>, // [H0, W0, C], kept on the host for bicubic resampling
    pos_hw: (usize, usize),
    layers: Vec<EncoderLayer>,
    final_ln: LayerNorm,
    hidden: usize,
    head_dim: usize,
    merge: [usize; 2],
    device: Device,
    dtype: DType,
}

impl MoonVit {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let ps = cfg.patch_size;
        let pe = vb.pp("patch_embed");
        let patch_w = pe.get((h, 3, ps, ps), "proj.weight")?.reshape((h, 3 * ps * ps))?.t()?.contiguous()?;
        let patch_b = pe.get(h, "proj.bias")?;
        let pos_hw = (cfg.init_pos_emb_height, cfg.init_pos_emb_width);
        let pos_emb = pe
            .get((pos_hw.0, pos_hw.1, h), "pos_emb.weight")?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let enc = vb.pp("encoder");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| EncoderLayer::new(cfg, enc.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let final_ln = layer_norm(h, LayerNormConfig { eps: LN_EPS, ..Default::default() }, enc.pp("final_layernorm"))?;
        Ok(Self {
            patch_w,
            patch_b,
            pos_emb,
            pos_hw,
            layers,
            final_ln,
            hidden: h,
            head_dim: h / cfg.num_attention_heads,
            merge: cfg.merge_kernel_size,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// `patches`: `[N, 3*P*P]` host data; returns merged features `[N/4, 4*C]`.
    pub fn forward(&self, patches: &[f32], grid_hw: (usize, usize)) -> Result<Tensor> {
        let (gh, gw) = grid_hw;
        let n = gh * gw;
        let x = Tensor::from_slice(patches, (n, patches.len() / n), &self.device)?.to_dtype(self.dtype)?;
        let x = x.matmul(&self.patch_w)?.broadcast_add(&self.patch_b)?;

        let pos = if (gh, gw) == self.pos_hw {
            self.pos_emb.clone()
        } else {
            bicubic_resize_hwc(&self.pos_emb, self.pos_hw, (gh, gw), self.hidden)
        };
        let pos = Tensor::from_vec(pos, (n, self.hidden), &self.device)?.to_dtype(self.dtype)?;
        let mut x = (x + pos)?;

        let (cos, sin) = rope_2d(gh, gw, self.head_dim, &self.device)?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }
        let x = self.final_ln.forward(&x)?;

        // patch_merger: [(h/kh) kh (w/kw) kw C] -> [(h/kh)(w/kw), kh*kw*C]
        let [kh, kw] = self.merge;
        x.reshape((gh / kh, kh, gw / kw, kw, self.hidden))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape(((gh / kh) * (gw / kw), kh * kw * self.hidden))
    }
}

/// cos/sin tables `[H*W, D/2]` for MoonViT's interleaved 2D rope: pair `2i`
/// rotates by `x * f_i`, pair `2i+1` by `y * f_i`, with `f_i = θ^(-4i/D)`.
fn rope_2d(h: usize, w: usize, dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..dim / 4)
        .map(|i| (1.0 / ROPE_THETA.powf((4 * i) as f64 / dim as f64)) as f32)
        .collect();
    let mut cos = Vec::with_capacity(h * w * half);
    let mut sin = Vec::with_capacity(h * w * half);
    for y in 0..h {
        for x in 0..w {
            for f in &freqs {
                for p in [x as f32, y as f32] {
                    let a = p * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (h * w, half), device)?,
        Tensor::from_vec(sin, (h * w, half), device)?,
    ))
}

/// Matches `F.interpolate(mode="bicubic", align_corners=False)` (A = -0.75),
/// applied separably on an `[H, W, C]` buffer.
fn bicubic_resize_hwc(src: &[f32], (ih, iw): (usize, usize), (oh, ow): (usize, usize), c: usize) -> Vec<f32> {
    fn taps(in_size: usize, out_size: usize) -> Vec<([usize; 4], [f32; 4])> {
        const A: f32 = -0.75;
        let cc1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
        let cc2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
        let scale = in_size as f32 / out_size as f32;
        (0..out_size)
            .map(|o| {
                let s = (o as f32 + 0.5) * scale - 0.5;
                let i0 = s.floor();
                let t = s - i0;
                let idx = [-1i64, 0, 1, 2].map(|d| (i0 as i64 + d).clamp(0, in_size as i64 - 1) as usize);
                (idx, [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)])
            })
            .collect()
    }

    let tw = taps(iw, ow);
    let mut tmp = vec![0f32; ih * ow * c];
    for y in 0..ih {
        for (x, (idx, wt)) in tw.iter().enumerate() {
            let dst = &mut tmp[(y * ow + x) * c..][..c];
            for k in 0..4 {
                let s = &src[(y * iw + idx[k]) * c..][..c];
                for (d, v) in dst.iter_mut().zip(s) {
                    *d += wt[k] * v;
                }
            }
        }
    }
    let th = taps(ih, oh);
    let mut out = vec![0f32; oh * ow * c];
    for (y, (idx, wt)) in th.iter().enumerate() {
        for k in 0..4 {
            let s = &tmp[idx[k] * ow * c..][..ow * c];
            let dst = &mut out[y * ow * c..][..ow * c];
            for (d, v) in dst.iter_mut().zip(s) {
                *d += wt[k] * v;
            }
        }
    }
    out
}

/// `mlp1`: LayerNorm -> Linear -> GELU(erf) -> Linear.
pub struct Projector {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Projector {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(in_dim, LayerNormConfig { eps: LN_EPS, ..Default::default() }, vb.pp("0"))?,
            fc1: linear(in_dim, out_dim, vb.pp("1"))?,
            fc2: linear(out_dim, out_dim, vb.pp("3"))?,
        })
    }
}

impl Module for Projector {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(&self.ln.forward(x)?)?.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

