#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function lint() {
  local path="$1"

  benchmark_runner::log::infoinfo "linting ${path}"
  uv run pre-commit run --all-files --show-diff-on-failure
}

#
# main
#

benchmark_runner::log::infoinfo "+++ LINT +++"
lint "benchmark_runner"
benchmark_runner::log::infoinfo "--- LINT ---"
