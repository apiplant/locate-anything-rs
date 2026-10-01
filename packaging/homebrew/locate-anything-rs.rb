# Generated from packaging/homebrew/locate-anything-rs.rb in apiplant/locate-anything-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/locate-anything-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class LocateAnythingRs < Formula
  desc "Rust (candle) inference for nvidia/LocateAnything-3B visual grounding"
  homepage "https://github.com/apiplant/locate-anything-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # No bottles: the release archives *are* the binaries, so the formula only
  # unpacks what the tagged workflow already built for each platform.
  on_macos do
    on_arm do
      url "https://github.com/apiplant/locate-anything-rs/releases/download/v@VERSION@/locate-anything-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/locate-anything-rs/releases/download/v@VERSION@/locate-anything-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/locate-anything-rs/releases/download/v@VERSION@/locate-anything-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  conflicts_with "locate-anything-rs-cuda", because: "both install the same binaries"

  def install
    bin.install "locate-anything"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/locate-anything --version")
  end
end
