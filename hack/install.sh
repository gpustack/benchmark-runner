#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory and third_party directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  if ! command -v uv &> /dev/null; then
    pip install uv
  fi
  # uv sync --all-extras to install all dependencies
  uv sync --locked
  if [[ "${DEPS_ONLY:-false}" == "false" ]]; then
    uv pip install pre-commit==3.7.1
    uv run pre-commit install
  fi
}

#
# main
#

guidellm_box::log::infoinfo "+++ DEPENDENCIES +++"
download_deps
guidellm_box::log::infoinfo "--- DEPENDENCIES ---"
