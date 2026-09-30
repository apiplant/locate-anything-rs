use candle_core::{DType, Result, Tensor};

/// Whether FlashAttention-2 kernels can run on this tensor (feature `flash-attn`,
/// CUDA device, half precision).
pub fn flash_usable(t: &Tensor) -> bool {
    cfg!(feature = "flash-attn") && t.device().is_cuda() && matches!(t.dtype(), DType::BF16 | DType::F16)
}

/// FlashAttention-2 over `[1, Q, H, D]` queries and `[1, T, Hkv, D]` keys/values
/// (bottom-right aligned when `causal`). Returns `[1, Q, H, D]`.
#[cfg(feature = "flash-attn")]
pub fn flash(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, scale as f32, causal)
}

#[cfg(not(feature = "flash-attn"))]
pub fn flash(_q: &Tensor, _k: &Tensor, _v: &Tensor, _scale: f64, _causal: bool) -> Result<Tensor> {
    candle_core::bail!("built without the `flash-attn` feature")
}

/// Upper bound on the number of attention scores materialized at once;
/// queries are processed in chunks to stay under it.
const SCORE_BUDGET: usize = 1 << 28;

/// Scaled dot-product attention over query chunks, with grouped-query support.
///
/// * `q`: `[G, R, Q, D]` — `G` kv groups, `R` query heads per group
/// * `k`, `v`: `[G, T, D]` (may be strided views, e.g. into a KV cache)
/// * `mask`: optional additive mask (0 / -inf) in `q`'s dtype, either `[Q, T]`
///   (broadcast over heads) or pre-expanded `[G*R, Q, T]`
///
/// Scores stay in the model dtype; candle's softmax kernel accumulates in f32.
/// Returns `[G, R, Q, D]`.
pub fn chunked_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, mask: Option<&Tensor>) -> Result<Tensor> {
    let (g, r, nq, d) = q.dims4()?;
    let t = k.dim(1)?;
    let q = (q * scale)?;
    let kt = k.t()?;
    let chunk = (SCORE_BUDGET / (g * r * t).max(1)).clamp(1, nq);

    let mut outs = Vec::with_capacity(nq.div_ceil(chunk));
    let mut q0 = 0;
    while q0 < nq {
        let c = chunk.min(nq - q0);
        let qc = q.narrow(2, q0, c)?.contiguous()?.reshape((g, r * c, d))?;
        let mut scores = qc.matmul(&kt)?;
        match mask {
            Some(m) if m.rank() == 3 && c == nq => scores = (scores + m.reshape((g, r * c, t))?)?,
            Some(m) => {
                let m = if m.rank() == 3 {
                    m.narrow(1, q0, c)?.reshape((g, r, c, t))?
                } else {
                    m.narrow(0, q0, c)?.reshape((1, 1, c, t))?
                };
                scores = scores.reshape((g, r, c, t))?.broadcast_add(&m)?.reshape((g, r * c, t))?;
            }
            None => {}
        }
        let p = candle_nn::ops::softmax_last_dim(&scores)?;
        outs.push(p.matmul(v)?.reshape((g, r, c, d))?);
        q0 += c;
    }
    if outs.len() == 1 {
        return Ok(outs.pop().unwrap());
    }
    Tensor::cat(&outs, 2)
}
