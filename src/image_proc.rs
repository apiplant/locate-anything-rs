//! Port of `LocateAnythingImageProcessor` (image_processing_locateanything.py).

use crate::config::PreprocessorConfig;
use anyhow::{bail, Result};
use image::{imageops::FilterType, DynamicImage, RgbImage};

pub struct ProcessedImage {
    /// Patches laid out as `[num_patches, 3, patch, patch]`, normalized.
    pub patches: Vec<f32>,
    /// Patch grid `(h, w)` fed to the vision tower.
    pub grid_hw: (usize, usize),
    /// Number of `<IMG_CONTEXT>` tokens after the 2x2 patch merge.
    pub num_tokens: usize,
}

/// Mirrors `to_rgb` in processing_locateanything.py: alpha is composited onto white.
pub fn to_rgb(img: &DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (o, p) in out.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as f32 / 255.0;
        for c in 0..3 {
            o[c] = (p[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        }
    }
    out
}

pub fn preprocess(img: &RgbImage, cfg: &PreprocessorConfig, in_token_limit: usize) -> Result<ProcessedImage> {
    let ps = cfg.patch_size as u32;
    let mut img = img.clone();
    let (w, h) = img.dimensions();

    let n = ((w / ps) * (h / ps)) as usize;
    if n > in_token_limit {
        let scale = (in_token_limit as f64 / n as f64).sqrt();
        let (nw, nh) = ((w as f64 * scale) as u32, (h as f64 * scale) as u32);
        img = image::imageops::resize(&img, nw.max(1), nh.max(1), FilterType::CatmullRom);
    }

    let (nw, nh) = img.dimensions();
    let pad_h = cfg.merge_kernel_size[0] as u32 * ps;
    let pad_w = cfg.merge_kernel_size[1] as u32 * ps;
    let tw = nw.div_ceil(pad_w) * pad_w;
    let th = nh.div_ceil(pad_h) * pad_h;
    if tw != nw || th != nh {
        img = image::imageops::resize(&img, tw, th, FilterType::CatmullRom);
    }

    let (w, h) = img.dimensions();
    if w / ps >= 512 || h / ps >= 512 {
        bail!("image exceeds the vision positional embedding range ({w}x{h})");
    }

    let (gh, gw) = ((h / ps) as usize, (w / ps) as usize);
    let ps = ps as usize;
    let mut patches = vec![0f32; gh * gw * 3 * ps * ps];
    let raw = img.as_raw();
    for py in 0..gh {
        for px in 0..gw {
            let base = (py * gw + px) * 3 * ps * ps;
            for c in 0..3 {
                let (mean, std) = (cfg.image_mean[c], cfg.image_std[c]);
                for iy in 0..ps {
                    for ix in 0..ps {
                        let (y, x) = (py * ps + iy, px * ps + ix);
                        let v = raw[(y * w as usize + x) * 3 + c] as f32 / 255.0;
                        patches[base + (c * ps + iy) * ps + ix] = (v - mean) / std;
                    }
                }
            }
        }
    }

    let merge = cfg.merge_kernel_size[0] * cfg.merge_kernel_size[1];
    Ok(ProcessedImage { patches, grid_hw: (gh, gw), num_tokens: gh * gw / merge })
}
