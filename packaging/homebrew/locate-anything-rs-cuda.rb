# Generated from packaging/homebrew/locate-anything-rs-cuda.rb in apiplant/locate-anything-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/locate-anything-rs-cuda.rb. Changes belong in
# the source repository: the next release overwrites this file.
class LocateAnythingRsCuda < Formula
  desc "Rust (candle) inference for nvidia/LocateAnything-3B visual grounding (CUDA build)"
  homepage "https://github.com/apiplant/locate-anything-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # Linux x86_64 only: no CUDA on Apple Silicon, and no arm64 CUDA build. It
  # needs an NVIDIA driver (libcuda) installed on the host, which Homebrew
  # cannot provide.
  depends_on :linux
  depends_on arch: :x86_64
  conflicts_with "locate-anything-rs", because: "both install the same binaries"

  url "https://github.com/apiplant/locate-anything-rs/releases/download/v@VERSION@/locate-anything-rs-cuda-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
  sha256 "@SHA_LINUX_X86_64_CUDA@"

  def install
    bin.install "locate-anything"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/locate-anything --version")
  end
end
