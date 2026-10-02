# locate-anything-rs

Rust ([candle](https://github.com/huggingface/candle)) inference for
[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B):
MoonViT vision encoder → MLP projector → Qwen2.5-3B, with **Parallel Box
Decoding** (6-token MTP blocks), plus the `fast` / `slow` / `hybrid` generation modes.

No Python needed at runtime. The tokenizer is built from `vocab.json`,
`merges.txt`, and `tokenizer_config.json`, and its output matches the HF tokenizer.

## Build

```sh
cargo build --release                         # CPU only (f32, slow)
cargo build --release --features cuda         # CUDA (bf16)
cargo build --release --features flash-attn   # CUDA + FlashAttention-2 (first build ~9 min)
```

`flash-attn` routes the vision tower and the LM prefill through
FlashAttention-2. Decode steps stay on the masked path because they have only a few queries.
The feature matters most for large images.

## Performance (RTX 4090, bf16, greedy, hybrid mode, end-to-end incl. vision)

| input | Python ref (warm, SDPA) | Rust | Rust + flash-attn |
| --- | ---: | ---: | ---: |
| 560x430 crop, 1 box | 0.14 s | 0.15 s | 0.15 s |
| 560x280 crop, 25 birds | 0.86 s | 0.59 s | 0.63 s |
| 835x320 crop, ~98 cars | 4.46 s | 3.44 s | 3.09 s |
| 2048x1206 full image | OOM (24 GB) | 2.25 s | 0.68 s |

The Rust numbers come from a cold process, and step counts vary slightly between
builds because bf16 rounding differs. A decode step costs about 10 ms. About 7 ms of that is
cuBLAS streaming the 6.2 GB of weights, which is close to the bandwidth limit. The rest
is candle's per-op launch and allocation overhead.

## Usage

```sh
# detection (comma-separated categories -> joined with </c>)
locate-anything --image street.jpg --task detect -q "person, car, bicycle" --draw out.png

# referring expression / pointing / OCR / GUI
locate-anything --image img.jpg --task ground -q "people wearing red shirts"
locate-anything --image img.jpg --task point  -q "the traffic light"
locate-anything --image img.jpg --task detect-text
locate-anything --image ui.png  --task gui    -q "the search button"

# JSON output, greedy, pure-AR decoding, step trace
locate-anything --image img.jpg -q "ship" --json --temperature 0 --mode slow -v
```

Without `--model`, the checkpoint is downloaded from Hugging Face on first use (about 7.6 GB) into
`$XDG_CACHE_HOME/locate-anything-rs/LocateAnything-3B` (default `~/.cache/locate-anything-rs/...`), and reused
afterwards. Pass `--model DIR` to use a local clone instead. Sampling defaults
match the upstream worker: `temperature=0.7, top_p=0.9, repetition_penalty=1.1`.
`--max-patches` lowers the vision budget (default 25600 patches ≈ 6400 LLM
tokens) for faster runs on big images.

## Parity

Checked against the official Python code (transformers 4.57.1, greedy decoding):
in f32 (`--f32` vs `torch_dtype=float32`), a 25-box hybrid-mode bird detection
matches **byte-for-byte**. That includes the MTP→AR fallback steps. In bf16,
the outputs share the same structure and step counts. Coordinates differ by a few
units out of 1000 because of kernel rounding, and the resize filter is not
identical to PIL's.

## Layout

| file | ports |
| --- | --- |
| `src/image_proc.rs` | `image_processing_locateanything.py` (resize to multiple of 28, patchify) |
| `src/vision.rs` | `modeling_vit.py` (MoonViT: bicubic pos-emb interp, 2D rope, patch merger) + `mlp1` |
| `src/qwen2.rs` | `modeling_qwen2.py` + `mask_magi_utils.py` (block-diffusion MTP window mask, KV cache) |
| `src/sampling.rs` | `generate_utils.py` (rep. penalty, top-p/k, `decode_bbox_avg`, `decode_ref`, `handle_pattern`) |
| `src/model.rs` | `LocateAnythingForConditionalGeneration.generate` |
| `src/tokenizer.rs` | Qwen2 BPE tokenizer assembled for `tokenizers` |
| `src/output.rs` | `<ref>…</ref><box><x1><y1><x2><y2></box>` → pixel boxes/points, drawing |

Attention is plain chunked SDPA (the score matrix is kept to about 1 GiB per chunk), so no
flash-attn or MagiAttention build is needed. Batch size is 1, same as upstream `generate`.

## Use as a library

```toml
[dependencies]
locate-anything = "0.1"                           # CPU
# locate-anything = { version = "0.1", features = ["cuda"] }   # + CUDA (opt-in; needs the CUDA toolkit to build)
```

GPU support is never on by default: depending on the crate never pulls in a CUDA toolchain. Enable `cuda`
(or `flash-attn`, which implies it) from your own `Cargo.toml` and the feature is passed down to candle.

```rust
use candle_core::{DType, Device};
use locate_anything::{prompts, GenerateOptions, LocateAnything};

// Downloads the checkpoint into ~/.cache/locate-anything-rs on first use (about 7.6 GB),
// or use `LocateAnything::load(path, &device, dtype)` for a local clone.
let device = Device::cuda_if_available(0)?;                       // Cpu without the `cuda` feature
let mut model = LocateAnything::from_pretrained(&device, DType::BF16)?;   // DType::F32 on CPU

let image = image::open("street.jpg")?.to_rgb8();
let found = model.locate(&image, &prompts::detect(&["person", "car"]), &GenerateOptions::greedy())?;
for d in &found.detections {
    println!("{d:?}");          // Detection::Box { label, x1, y1, x2, y2 } / Detection::Point { .. }
}
```

`locate_anything::prompts` has one builder per task (`detect`, `ground`, `ground_single`, `text`,
`detect_text`, `gui`, `point`); `output::draw` paints detections onto an image; `preprocess`,
`encode_image`, `build_prompt` and `generate` are public for callers that drive the stages themselves.

## Install

Prebuilt packages (locate-anything) for macOS (Apple Silicon), Linux x86_64 and Linux arm64:

```bash
brew tap apiplant/tap && brew install apiplant/tap/locate-anything-rs      # macOS, Linux
sudo apt install locate-anything-rs      # Debian/Ubuntu, after adding apt.apiplant.com
sudo pacman -S locate-anything-rs        # Arch, after adding apiplant.github.io/pacman
```

CUDA builds (Linux x86_64, NVIDIA GPU) are separate packages: `locate-anything-rs-cuda` (`brew install apiplant/tap/locate-anything-rs-cuda`, `sudo apt install locate-anything-rs-cuda`, `sudo pacman -S locate-anything-rs-cuda`). They conflict with `locate-anything-rs`.

Setup commands for the apt and pacman repositories, the plain archives and the release process are in [`packaging/README.md`](packaging/README.md). Release archives are on the [releases page](https://github.com/apiplant/locate-anything-rs/releases).

Website and in-browser demo: <https://locate-anything-rs.apiplant.com>.
