use anyhow::{bail, Result};
use candle_core::{DType, Device};
use clap::{Parser, ValueEnum};
use locate_anything::{image_proc, output, GenerateOptions, GenerationMode, LocateAnything, SamplingParams};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    /// Object detection / layout analysis; --query is a comma-separated category list.
    Detect,
    /// Phrase grounding, all matching instances.
    Ground,
    /// Phrase grounding, a single instance.
    GroundSingle,
    /// Locate a piece of text.
    Text,
    /// Scene text detection (no query).
    DetectText,
    /// GUI element grounding (box).
    Gui,
    /// Pointing / GUI grounding (point).
    Point,
    /// Send --query verbatim as the prompt.
    Raw,
}

impl Task {
    fn prompt(self, q: &str) -> String {
        use locate_anything::prompts;
        match self {
            Task::Detect => prompts::detect(&q.split(',').collect::<Vec<_>>()),
            Task::Ground => prompts::ground(q),
            Task::GroundSingle => prompts::ground_single(q),
            Task::Text => prompts::text(q),
            Task::DetectText => prompts::detect_text(),
            Task::Gui => prompts::gui(q),
            Task::Point => prompts::point(q),
            Task::Raw => q.into(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about = "LocateAnything-3B visual grounding in Rust (candle)")]
struct Args {
    /// Model directory (a clone of nvidia/LocateAnything-3B). Defaults to the checkpoint in the
    /// locate-anything-rs cache directory (`$XDG_CACHE_HOME/locate-anything-rs`, else
    /// `~/.cache/locate-anything-rs`), downloaded there from Hugging Face first if it isn't present.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    image: PathBuf,
    #[arg(long, value_enum, default_value = "detect")]
    task: Task,
    #[arg(long, short, default_value = "")]
    query: String,
    #[arg(long, value_enum, default_value = "hybrid")]
    mode: GenerationMode,
    #[arg(long, default_value_t = 8192)]
    max_new_tokens: usize,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Max vision patches (14x14) before downscaling; defaults to preprocessor_config (25600).
    #[arg(long)]
    max_patches: Option<usize>,
    /// Write a copy of the image with detections drawn on it.
    #[arg(long)]
    draw: Option<PathBuf>,
    /// Print detections as JSON on stdout (instead of the human-readable summary).
    #[arg(long)]
    json: bool,
    #[arg(long)]
    cpu: bool,
    /// Run in f32 instead of bf16 on GPU (slower, for numerical checks).
    #[arg(long)]
    f32: bool,
    /// Print each decoding step (mtp/ar) to stderr.
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.query.is_empty() && !matches!(args.task, Task::DetectText) {
        bail!("--query is required for task {:?}", args.task);
    }

    let model_dir = match &args.model {
        Some(dir) => dir.clone(),
        None => locate_anything::download::download()?,
    };

    let device = if args.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    let dtype = if device.is_cpu() || args.f32 { DType::F32 } else { DType::BF16 };

    let t = Instant::now();
    let mut model = LocateAnything::load(&model_dir, &device, dtype)?;
    eprintln!("loaded model on {device:?} ({dtype:?}) in {:.1}s", t.elapsed().as_secs_f64());

    let img = image_proc::to_rgb(&image::open(&args.image)?);
    let processed = model.preprocess(&img, args.max_patches)?;
    eprintln!(
        "image {}x{} -> patch grid {:?}, {} visual tokens",
        img.width(),
        img.height(),
        processed.grid_hw,
        processed.num_tokens
    );

    let prompt = args.task.prompt(&args.query);
    let opts = GenerateOptions {
        mode: args.mode,
        max_new_tokens: args.max_new_tokens,
        sampling: SamplingParams {
            temperature: args.temperature,
            top_p: Some(args.top_p),
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
        },
        seed: args.seed,
        verbose: args.verbose,
    };
    let (answer, stats) = model.generate(&processed, &prompt, &opts)?;
    let dets = output::parse(&answer, img.width(), img.height());

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "prompt": prompt, "answer": answer, "detections": dets, "stats": stats,
            }))?
        );
    } else {
        println!("{answer}");
        for d in &dets {
            println!("{}", serde_json::to_string(d)?);
        }
    }
    let n_boxes = answer.matches("<box>").count();
    eprintln!(
        "tokens={} steps={} boxes={} switch_to_ar={} vision={:.3}s prefill={:.3}s decode_fwd={:.3}s sampling={:.3}s total={:.3}s tps={:.1} bps={:.1}",
        stats.num_tokens,
        stats.forward_steps,
        n_boxes,
        stats.switch_to_ar,
        stats.vision_secs,
        stats.prefill_secs,
        stats.forward_secs,
        stats.sampling_secs,
        stats.total_secs,
        stats.num_tokens as f64 / stats.total_secs,
        n_boxes as f64 / stats.total_secs,
    );

    if let Some(path) = &args.draw {
        let mut canvas = img.clone();
        output::draw(&mut canvas, &dets);
        canvas.save(path)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
