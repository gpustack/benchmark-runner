
ARG BENCHMARK_RUNNER_BASE_IMAGE=base

# Stage Base

FROM ubuntu:22.04@sha256:3c61d3759c2639d4b836d32a2d3c83fa0214e36f195a3421018dbaaf79cbe37f AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG PYTHON_VERSION=3.11

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Install Tools

ENV DEBIAN_FRONTEND=noninteractive \
    LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'

RUN <<EOF
    # Tools

    # Refresh
    apt-get update -y && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates gnupg2 lsb-release \
      && apt-get update -y

    # Install
    apt-get install -y --no-install-recommends \
        ca-certificates build-essential bash openssl \
        curl wget \
        git \
        tzdata locales \
        vim

    # Update locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

    # Update timezone
    rm -f /etc/localtime \
        && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
        && echo "Asia/Shanghai" > /etc/timezone \
        && dpkg-reconfigure --frontend noninteractive tzdata

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF

## Upgrade Python if needed

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN <<EOF
    # Python

    if (( $(echo "$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2) == ${PYTHON_VERSION}" | bc -l) )); then
        echo "Skipping Python upgrade for ${PYTHON_VERSION}..."
        if [[ -z "$(ldconfig -v 2>/dev/null | grep libpython${PYTHON_VERSION})" ]]; then
            PYTHON_LIB_PREFIX=$(python3 -c "import sys; print(sys.base_prefix);")
            echo "${PYTHON_LIB_PREFIX}/lib" >> /etc/ld.so.conf.d/python3.conf
            echo "${PYTHON_LIB_PREFIX}/lib64" >> /etc/ld.so.conf.d/python3.conf
            ldconfig -v
        fi
        exit 0
    fi

    # Add deadsnakes PPA for Python versions
    for i in 1 2 3; do
        add-apt-repository -y ppa:deadsnakes/ppa && break || { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }
    done
    apt-get update -y

    # Install
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils

    # Update alternatives
    if [[ -f /etc/alternatives/python3 ]]; then update-alternatives --remove-all python3; fi; update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    if [[ -f /etc/alternatives/python ]]; then update-alternatives --remove-all python; fi; update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
    curl -sS "https://bootstrap.pypa.io/get-pip.py" | python${PYTHON_VERSION}

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /var/cache/apt
EOF


## Install Python buildkit

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIPX_HOME=/root/.local/share/pipx \
    PIPX_LOCAL_VENVS=/root/.local/share/pipx/venvs \
    POETRY_NO_CACHE=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY="unsafe-best-match"

RUN <<EOF
    # Buildkit

    cat <<EOT >/tmp/requirements.txt
setuptools
wheel
hatchling
uv
py-spy
EOT
    pip install -r /tmp/requirements.txt

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/*
EOF

# Stage benchmark-runner

FROM ${BENCHMARK_RUNNER_BASE_IMAGE} AS benchmark-runner
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

## Step: Download ShareGPT dataset
RUN mkdir -p /workspace/benchmark-runner/sharegpt_data \
    && cd /workspace/benchmark-runner/sharegpt_data \
    && FILE=ShareGPT_V3_unfiltered_cleaned_split.json \
    && if [ ! -f "$FILE" ]; then \
        wget --progress=bar:force:noscroll -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$FILE; \
    else \
        echo "$FILE already exists."; \
    fi

## Step: Install Benchmark Runner
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,target=/workspace/benchmark-runner,rw \
    --mount=type=bind,from=ghcr.io/oras-project/oras:v1.3.0,source=/bin/oras,dst=/bin/oras <<EOF
    set -e
    export POETRY_NO_CACHE=0
    export UV_NO_CACHE=0
    export UV_SYSTEM_PYTHON=1
    export UV_LINK_MODE=copy

    cd /workspace/benchmark-runner \
        && git config --global --add safe.directory /workspace/benchmark-runner \
        && make build

    # Install Benchmark Runner.
    WHEEL_PACKAGE="$(ls /workspace/benchmark-runner/dist/*.whl)[all]";
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu/ \
        ${WHEEL_PACKAGE}

    # Review
    uv pip tree \
        --package benchmark-runner
    benchmark-runner --version

    # Cleanup
    rm -rf /var/tmp/* \
        && rm -rf /tmp/* \
        && rm -rf /workspace/benchmark-runner/dist
EOF

ENTRYPOINT [ "benchmark-runner" ]
CMD [ "benchmark", "run" ]
