#!/usr/bin/env bash
# Install the ACE-Step 1.5 backend into the current Python environment (Python 3.11 or 3.12).
#
# ACE-Step pins CUDA-specific torch builds and a bundled `nano-vllm` package that are normally
# resolved by `uv`, so a plain `pip install git+...` fails. This script does the equivalent with pip.
set -euo pipefail

ACESTEP_DIR="${ACESTEP_DIR:-third_party/ACE-Step-1.5}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -d "$ACESTEP_DIR" ]; then
  git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git "$ACESTEP_DIR"
fi

if [ "$(uname -s)" = "Linux" ]; then
  # Bundled dependency that is not on PyPI; install it first so pip treats it as satisfied.
  pip install --extra-index-url "$TORCH_INDEX" "$ACESTEP_DIR/acestep/third_parts/nano-vllm"
fi
pip install --extra-index-url "$TORCH_INDEX" -e "$ACESTEP_DIR"
