import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>locate-anything</IC> crate is the engine behind the CLI: model loading, image preprocessing,
        generation with Parallel Box Decoding, and output parsing.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <CopyBlock command={`cargo add locate-anything --git https://github.com/apiplant/locate-anything-rs
# CPU only, no CUDA toolchain needed:
cargo add locate-anything --git https://github.com/apiplant/locate-anything-rs --no-default-features`} />
      </Section>

      <Section>
        <H2>Detect objects</H2>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::{DType, Device};
use locate_anything::{image_proc, output, GenerateOptions, GenerationMode, LocateAnything, SamplingParams};

let device = Device::cuda_if_available(0)?;
let mut model = LocateAnything::load("LocateAnything-3B".as_ref(), &device, DType::BF16)?;

let img = image_proc::to_rgb(&image::open("street.jpg")?);
let processed = model.preprocess(&img, None)?;          // None = the checkpoint's patch budget

let prompt = "Locate all the instances that matches the following description: person</c>car.";
let opts = GenerateOptions {
    mode: GenerationMode::Hybrid,
    max_new_tokens: 8192,
    sampling: SamplingParams { temperature: 0.0, top_p: None, top_k: None, repetition_penalty: 1.1 },
    seed: 0,
    verbose: false,
};
let (answer, stats) = model.generate(&processed, prompt, &opts)?;

for d in output::parse(&answer, img.width(), img.height()) {
    println!("{d:?}");
}
println!("{} forward steps, {:.2}s", stats.forward_steps, stats.total_secs);`}</Pre>
        <P>
          <IC>output::parse</IC> returns <IC>Detection::Box</IC> or <IC>Detection::Point</IC> values in pixel
          coordinates; <IC>output::draw</IC> paints them onto an <IC>RgbImage</IC>. Use <IC>DType::F32</IC> on
          the CPU.
        </P>
      </Section>

      <Section>
        <H2>Pieces</H2>
        <P>
          <IC>model.encode_image</IC> runs the vision tower alone and <IC>model.build_prompt</IC> tokenizes a
          question with the right number of image tokens: both are public, for callers that want to drive the
          stages themselves.
        </P>
      </Section>
    </DocsLayout>
  );
}
