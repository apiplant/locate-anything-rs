//! Rust ([candle](https://github.com/huggingface/candle)) inference for
//! [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B): open-vocabulary
//! detection, referring-expression grounding, pointing, OCR and GUI grounding, with Parallel Box
//! Decoding.
//!
//! ```no_run
//! use candle_core::{DType, Device};
//! use locate_anything::{prompts, GenerateOptions, LocateAnything};
//!
//! // Downloads the checkpoint into ~/.cache/locate-anything-rs on first use (about 7.6 GB).
//! let device = Device::cuda_if_available(0)?;
//! let mut model = LocateAnything::from_pretrained(&device, DType::BF16)?;
//!
//! let image = image::open("street.jpg")?.to_rgb8();
//! let found = model.locate(&image, &prompts::detect(&["person", "car"]), &GenerateOptions::greedy())?;
//! for d in &found.detections {
//!     println!("{d:?}");
//! }
//! # anyhow::Ok(())
//! ```

pub mod attention;
pub mod config;
pub mod download;
pub mod image_proc;
pub mod model;
pub mod output;
pub mod prompts;
pub mod qwen2;
pub mod sampling;
pub mod tokenizer;
pub mod vision;

pub use model::{GenerateOptions, GenerateStats, Located, LocateAnything};
pub use output::Detection;
pub use sampling::{GenerationMode, SamplingParams};
