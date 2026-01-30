#!/usr/bin/env bash


function benchmark_runner::util::sed() {
  if ! sed -i "$@" >/dev/null 2>&1; then
    # back off none GNU sed
    sed -i "" "$@"
  fi
}

function benchmark_runner::util::get_os_name() {
  # Support overriding by BUILD_OS for cross-building
  local os_name="${BUILD_OS:-}"
  if [[ -n "$os_name" ]]; then
    echo "$os_name" | tr '[:upper:]' '[:lower:]'
  else
    uname -s | tr '[:upper:]' '[:lower:]'
  fi
}

function benchmark_runner::util::is_darwin() {
  [[ "$(benchmark_runner::util::get_os_name)" == "darwin" ]]
}

function benchmark_runner::util::is_linux() {
  [[ "$(benchmark_runner::util::get_os_name)" == "linux" ]]
}
