#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function ci() {
  make install "$@"
  make deps "$@"
  make lint "$@"
  make build "$@"
}

#
# main
#

benchmark_runner::log::infoinfo "+++ CI +++"
ci "$@"
benchmark_runner::log::infoinfo "--- CI ---"
