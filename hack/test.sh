#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function test() {
  uv run python -m pytest
}

#
# main
#

benchmark_runner::log::infoinfo "+++ TEST +++"
test
benchmark_runner::log::infoinfo "--- TEST ---"
