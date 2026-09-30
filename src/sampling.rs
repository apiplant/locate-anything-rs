//! Port of generate_utils.py: logits processing, sampling, and the Parallel
//! Box Decoding block interpretation (`decode_bbox_avg`, `decode_ref`,
//! `handle_pattern`).

use crate::config::TokenIds;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GenerationMode {
    /// MTP only, never falls back to auto-regressive decoding.
    Fast,
    /// Pure auto-regressive decoding.
    Slow,
    /// MTP first, AR fallback on malformed/uncertain boxes.
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}

/// Number of highest-probability candidates tracked per row. Top-p nuclei and
/// the top-k lookups of the block decoders almost always fit in this.
const TOP_N: usize = 64;

/// Post-processing distribution of one logits row.
pub struct RowProbs {
    /// Dense probabilities over the vocabulary.
    pub p: Vec<f32>,
    /// Token ids sorted by descending probability: every non-zero entry when
    /// `complete`, otherwise the `TOP_N` largest.
    top: Vec<u32>,
    complete: bool,
}

impl RowProbs {
    pub fn get(&self, tok: u32) -> f32 {
        self.p[tok as usize]
    }

    /// `torch.topk(p, k)` indices. Zero-probability slots (only possible after
    /// top-p/top-k filtering) are filled with the lowest unused token ids.
    pub fn top_k(&self, k: usize) -> Vec<u32> {
        let mut out: Vec<u32> = self.top.iter().copied().take(k).collect();
        if out.len() < k && self.complete {
            out.extend((0..self.p.len() as u32).filter(|t| self.p[*t as usize] == 0.0).take(k - out.len()));
        }
        out
    }
}

/// Applies repetition penalty, temperature, top-p and top-k to each logits row
/// (the order used by generate_utils.sample_tokens) and returns per-row
/// probabilities plus the sampled (or argmax) token.
pub fn sample_tokens<R: Rng>(
    rows: Vec<Vec<f32>>,
    seen: &[u32],
    p: &SamplingParams,
    rng: &mut R,
) -> (Vec<RowProbs>, Vec<u32>) {
    let probs: Vec<RowProbs> = if rows.len() == 1 {
        rows.into_iter().map(|r| process_row(r, seen, p)).collect()
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = rows.into_iter().map(|r| sc.spawn(move || process_row(r, seen, p))).collect();
            hs.into_iter().map(|h| h.join().expect("sampling thread panicked")).collect()
        })
    };
    let x0 = probs
        .iter()
        .map(|rp| if p.temperature > 0.0 { sample(rp, rng) } else { rp.top[0] })
        .collect();
    (probs, x0)
}

fn sample<R: Rng>(rp: &RowProbs, rng: &mut R) -> u32 {
    if rp.complete {
        let r: f32 = rng.gen::<f32>() * rp.top.iter().map(|&t| rp.get(t)).sum::<f32>();
        let mut acc = 0f32;
        for &t in &rp.top {
            acc += rp.get(t);
            if r < acc {
                return t;
            }
        }
        return *rp.top.last().unwrap_or(&0);
    }
    match WeightedIndex::new(&rp.p) {
        Ok(d) => d.sample(rng) as u32,
        Err(_) => rp.top[0],
    }
}

fn process_row(mut logits: Vec<f32>, seen: &[u32], p: &SamplingParams) -> RowProbs {
    if p.repetition_penalty != 1.0 {
        for &t in seen {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / p.repetition_penalty } else { *l * p.repetition_penalty };
            }
        }
    }
    if p.temperature > 0.0 {
        let inv = 1.0 / p.temperature;
        logits.iter_mut().for_each(|l| *l *= inv);
    }
    if p.top_k.is_some_and(|k| k > TOP_N) {
        return process_row_slow(logits, p);
    }

    // exp(l - max) in place; `logits` becomes unnormalized probabilities.
    // Terms below e^-30 cannot move an f32 sum over the vocabulary, so they
    // are flushed to zero without calling exp (most of the vocabulary).
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = m - 30.0;
    let mut z = 0f32;
    for l in logits.iter_mut() {
        *l = if *l > floor { (*l - m).exp() } else { 0.0 };
        z += *l;
    }
    let mut e = logits;
    let mut top = top_n(&e, TOP_N);

    let mut kept: Option<Vec<u32>> = None;
    if let Some(tp) = p.top_p.filter(|&tp| tp < 1.0) {
        // Widen the sorted candidate set until it covers the nucleus.
        let mut width = TOP_N;
        loop {
            let mut cum = 0f32;
            let cut = top.iter().position(|&t| {
                cum += e[t as usize] / z;
                cum > tp
            });
            if let Some(i) = cut {
                kept = Some(top[..=i].to_vec());
                break;
            }
            if width >= e.len() {
                // Rounding kept the cumulative sum at or below top_p: keep everything non-zero.
                kept = Some(top.iter().copied().filter(|&t| e[t as usize] > 0.0).collect());
                break;
            }
            width = (width * 32).min(e.len());
            top = top_sorted(&e, width);
        }
    }
    if let Some(k) = p.top_k {
        let base = kept.take().unwrap_or_else(|| top.clone());
        // Ties with the k-th value survive, as with `logits < kth` in torch.
        let kth = e[base[(k.min(base.len())).max(1) - 1] as usize];
        kept = Some(base.into_iter().filter(|&t| e[t as usize] >= kth).collect());
    }

    match kept {
        Some(kept) => {
            let zk: f32 = kept.iter().map(|&t| e[t as usize]).sum();
            let mut dense = vec![0f32; e.len()];
            for &t in &kept {
                dense[t as usize] = e[t as usize] / zk;
            }
            RowProbs { p: dense, top: kept, complete: true }
        }
        None => {
            e.iter_mut().for_each(|x| *x /= z);
            RowProbs { p: e, top, complete: false }
        }
    }
}

/// Exact fallback with full sorting, for wide nuclei or large top-k.
fn process_row_slow(mut logits: Vec<f32>, p: &SamplingParams) -> RowProbs {
    let order = |v: &[f32]| {
        let mut idx: Vec<u32> = (0..v.len() as u32).collect();
        idx.sort_by(|a, b| v[*b as usize].total_cmp(&v[*a as usize]));
        idx
    };
    if let Some(tp) = p.top_p.filter(|&tp| tp < 1.0) {
        let pr = softmax(&logits);
        let mut keep = vec![false; logits.len()];
        let mut cum = 0f32;
        for i in order(&pr) {
            keep[i as usize] = true;
            cum += pr[i as usize];
            if cum > tp {
                break;
            }
        }
        for (l, k) in logits.iter_mut().zip(keep) {
            if !k {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    if let Some(k) = p.top_k {
        let o = order(&logits);
        let kth = logits[o[k.min(o.len()).max(1) - 1] as usize];
        logits.iter_mut().filter(|l| **l < kth).for_each(|l| *l = f32::NEG_INFINITY);
    }
    let pr = softmax(&logits);
    let top: Vec<u32> = order(&pr).into_iter().take_while(|&t| pr[t as usize] > 0.0).collect();
    RowProbs { p: pr, top, complete: true }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
    let s: f32 = out.iter().sum();
    out.iter_mut().for_each(|x| *x /= s);
    out
}

/// Indices of the `n` largest values, sorted descending (partial selection).
fn top_sorted(v: &[f32], n: usize) -> Vec<u32> {
    let cmp = |a: &u32, b: &u32| v[*b as usize].total_cmp(&v[*a as usize]);
    let mut idx: Vec<u32> = (0..v.len() as u32).filter(|&i| v[i as usize] > 0.0).collect();
    if n < idx.len() {
        idx.select_nth_unstable_by(n, cmp);
        idx.truncate(n);
    }
    idx.sort_by(cmp);
    idx
}

/// Indices of the `n` largest values, sorted descending, in a single pass.
fn top_n(v: &[f32], n: usize) -> Vec<u32> {
    let mut buf: Vec<(f32, u32)> = Vec::with_capacity(n + 1);
    let mut floor = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if buf.len() == n && x <= floor {
            continue;
        }
        let pos = buf.partition_point(|&(y, _)| y >= x);
        buf.insert(pos, (x, i as u32));
        if buf.len() > n {
            buf.pop();
        }
        if buf.len() == n {
            floor = buf[n - 1].0;
        }
    }
    buf.into_iter().map(|(_, i)| i).collect()
}

enum BoxFrame {
    Empty,
    Legal,
    Illegal,
}

fn box_frame(probs: &[RowProbs], t: &TokenIds, start_thresh: f32, end_thresh: f32) -> BoxFrame {
    let at = |pos: usize, tok: u32| probs[pos].get(tok);
    if at(0, t.box_start) >= start_thresh
        && at(1, t.none) > 0.2
        && at(2, t.box_end) > 0.2
        && at(3, t.null) > 0.1
        && at(4, t.null) > 0.1
    {
        return BoxFrame::Empty;
    }
    if at(5, t.box_end) + at(5, t.null) + at(5, t.im_end) >= end_thresh {
        BoxFrame::Legal
    } else {
        BoxFrame::Illegal
    }
}

/// `decode_bbox_avg`: reads a 6-token block as `<box> c c c c </box>` using the
/// top-k coordinate candidates at each position.
fn decode_bbox(probs: &[RowProbs], t: &TokenIds, keep_k: usize, mode: GenerationMode) -> Option<Vec<u32>> {
    match box_frame(probs, t, 0.7, 0.2) {
        BoxFrame::Empty => return Some(vec![t.box_start, t.none, t.box_end, t.null, t.null, t.null]),
        BoxFrame::Illegal => return None,
        BoxFrame::Legal => {}
    }
    let mut coords = Vec::with_capacity(4);
    for row in &probs[1..5] {
        let cands: Vec<(u32, f32)> = row.top_k(keep_k).into_iter().map(|i| (i, row.get(i))).collect();
        let valid: Vec<&(u32, f32)> = cands.iter().filter(|(id, _)| t.is_coord(*id)).collect();
        let &&(first_id, first_p) = valid.first()?;
        let abnormal = mode == GenerationMode::Hybrid && first_p < 0.9 && valid.len() > 1 && {
            let max = valid.iter().map(|c| c.0).max().unwrap_or(0) as i64;
            let min = valid.iter().map(|c| c.0).min().unwrap_or(0) as i64;
            max - min > 60
        };
        // Abnormal (spatially ambiguous) coordinates become token 0, which is
        // not a coordinate and makes handle_pattern fall back to AR.
        coords.push(if abnormal { 0 } else { first_id });
    }
    let mut out = vec![t.box_start];
    out.extend(coords);
    out.push(t.box_end);
    Some(out)
}

/// `decode_ref`: reads a block as `<ref>` + label text tokens.
fn decode_ref(probs: &[RowProbs], t: &TokenIds, keep_k: usize, start_thresh: f32) -> Option<Vec<u32>> {
    if probs[0].get(t.ref_start) < start_thresh {
        return None;
    }
    let mut out = vec![t.ref_start];
    for row in &probs[1..] {
        let tok = row.top_k(keep_k).into_iter().find(|&i| !t.is_coord(i))?;
        out.push(tok);
    }
    Some(out)
}

/// Interprets the 6 MTP logit rows as a block of tokens.
pub fn decode_block(probs: &[RowProbs], x0: &[u32], t: &TokenIds, mode: GenerationMode) -> Vec<u32> {
    decode_bbox(probs, t, 4, mode)
        .or_else(|| decode_ref(probs, t, 5, 0.6))
        .unwrap_or_else(|| x0.to_vec())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    ImEnd,
    EmptyBox,
    CoordBox,
    PointBox,
    ErrorBox,
    RefObject,
}

/// `handle_pattern`: validates a decoded block and returns the tokens to commit.
pub fn handle_pattern(x0: &[u32], t: &TokenIds, mode: GenerationMode) -> (BlockType, Vec<u32>) {
    if x0[0] == t.null || x0[0] == t.im_end {
        return (BlockType::ImEnd, vec![t.im_end]);
    }
    if x0[..2] == [t.box_start, t.none] {
        return (BlockType::EmptyBox, vec![t.box_start, t.none, t.box_end]);
    }
    if x0[0] == t.box_start {
        let coord_ix = 1 + x0[1..5].iter().take_while(|&&c| t.is_coord(c)).count();
        if coord_ix == 5 && x0[5] == t.box_end {
            return (BlockType::CoordBox, x0.to_vec());
        }
        if coord_ix == 3 && x0[3] == t.box_end {
            return (BlockType::PointBox, x0[..4].to_vec());
        }
        return match mode {
            GenerationMode::Fast => (BlockType::CoordBox, x0.to_vec()),
            _ => (BlockType::ErrorBox, x0[..coord_ix].to_vec()),
        };
    }
    let mut toks: Vec<u32> = x0.iter().copied().take_while(|&x| x != t.null).collect();
    let n = toks.len();
    if n >= 2 && toks[n - 1] == t.ref_end && toks[n - 2] == t.ref_end {
        toks.pop();
    }
    (BlockType::RefObject, toks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> TokenIds {
        TokenIds {
            box_start: 151668,
            box_end: 151669,
            coord_start: 151677,
            coord_end: 152677,
            ref_start: 151672,
            ref_end: 151673,
            none: 4064,
            null: 152678,
            im_end: 151645,
            switch: 152679,
            mask: 151676,
        }
    }

    #[test]
    fn patterns() {
        let t = ids();
        let c = |v: u32| t.coord_start + v;
        let full = [t.box_start, c(1), c(2), c(3), c(4), t.box_end];
        assert_eq!(handle_pattern(&full, &t, GenerationMode::Hybrid), (BlockType::CoordBox, full.to_vec()));
        let point = [t.box_start, c(1), c(2), t.box_end, t.null, t.null];
        assert_eq!(handle_pattern(&point, &t, GenerationMode::Hybrid).0, BlockType::PointBox);
        let bad = [t.box_start, c(1), 0, c(3), c(4), t.box_end];
        assert_eq!(
            handle_pattern(&bad, &t, GenerationMode::Hybrid),
            (BlockType::ErrorBox, vec![t.box_start, c(1)])
        );
        assert_eq!(handle_pattern(&bad, &t, GenerationMode::Fast).0, BlockType::CoordBox);
        let r = [t.ref_start, 10, 11, t.ref_end, t.ref_end, t.null];
        assert_eq!(handle_pattern(&r, &t, GenerationMode::Hybrid), (BlockType::RefObject, vec![t.ref_start, 10, 11, t.ref_end]));
        assert_eq!(handle_pattern(&[t.null; 6], &t, GenerationMode::Hybrid).0, BlockType::ImEnd);
    }

    fn params(top_p: Option<f32>, top_k: Option<usize>) -> SamplingParams {
        SamplingParams { temperature: 0.7, top_p, top_k, repetition_penalty: 1.1 }
    }

    #[test]
    fn top_p_keeps_first_above_threshold() {
        let rp = process_row(vec![3.0, 2.0, 1.0, 0.0], &[], &params(Some(0.5), None));
        assert_eq!(rp.top, vec![0]);
        assert_eq!(rp.p, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(rp.top_k(3), vec![0, 1, 2]);
    }

    #[test]
    fn fast_path_matches_slow_path() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for (top_p, top_k) in [(Some(0.9), None), (None, None), (Some(0.95), Some(8)), (None, Some(20))] {
            for peaky in [0.5f32, 4.0, 12.0] {
                let logits: Vec<f32> = (0..5000).map(|_| rng.gen::<f32>() * peaky).collect();
                let seen = [3u32, 17, 4000];
                let pr = params(top_p, top_k);
                let fast = process_row(logits.clone(), &seen, &pr);
                let mut l2 = logits.clone();
                for &t in &seen {
                    let l = &mut l2[t as usize];
                    *l = if *l > 0.0 { *l / 1.1 } else { *l * 1.1 };
                }
                l2.iter_mut().for_each(|l| *l /= 0.7);
                let slow = process_row_slow(l2, &pr);
                assert_eq!(fast.top_k(5), slow.top_k(5));
                for (a, b) in fast.p.iter().zip(&slow.p) {
                    assert!((a - b).abs() < 1e-5, "{a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn top_n_single_pass() {
        let v = [0.1f32, 0.9, 0.3, 0.9, 0.5];
        assert_eq!(top_n(&v, 3), vec![1, 3, 4]);
    }
}
