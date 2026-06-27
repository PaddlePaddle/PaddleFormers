#!/bin/bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to download and install paddlefleet_ops wheel based on local git state.
# Automatically detects the current CUDA version. Only CUDA 13.0 and CUDA 12.9 are supported.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect CUDA version from the current environment
detect_cuda_version() {
    local cuda_ver=""
    # Try nvcc first
    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    # Fallback to nvidia-smi
    elif command -v nvidia-smi &>/dev/null; then
        cuda_ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    # Fallback to CUDA_HOME/CUDA_PATH
    elif [[ -n "${CUDA_HOME:-}" ]]; then
        cuda_ver=$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [[ -n "${CUDA_PATH:-}" ]]; then
        cuda_ver=$("${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    fi
    echo "$cuda_ver"
}

DETECTED_CUDA_VERSION=$(detect_cuda_version)
if [[ -z "$DETECTED_CUDA_VERSION" ]]; then
    print_error "Cannot detect CUDA version. Please ensure CUDA toolkit is installed (nvcc or nvidia-smi available)."
    exit 1
fi

print_info "Detected CUDA version: $DETECTED_CUDA_VERSION"

# Check if the detected CUDA version is supported
case "$DETECTED_CUDA_VERSION" in
    "13.2")
        CUDA_SUFFIX="cu132"
        ;;
    "13.0")
        CUDA_SUFFIX="cu130"
        ;;
    "12.9")
        CUDA_SUFFIX="cu129"
        ;;
    *)
        print_error "Unsupported CUDA version: $DETECTED_CUDA_VERSION"
        print_error "Only CUDA 13.2, CUDA 13.0 and CUDA 12.9 are supported."
        exit 1
        ;;
esac

print_info "Using CUDA version: $DETECTED_CUDA_VERSION (suffix: $CUDA_SUFFIX)"

# Get workspace root (assuming this script is run from the repository root)
WORKSPACE_ROOT="$(git rev-parse --show-toplevel)"
cd "$WORKSPACE_ROOT"

# Get base version from version.txt
VERSION_FILE="$WORKSPACE_ROOT/version.txt"
if [[ ! -f "$VERSION_FILE" ]]; then
    print_error "version.txt not found at $VERSION_FILE"
    exit 1
fi
BASE_VERSION=$(cat "$VERSION_FILE" | head -1 | tr -d '[:space:]')
print_info "Base version: $BASE_VERSION"

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

print_info "Current branch: $CURRENT_BRANCH"

# Find base branch (develop or release/*)
if [[ "$CURRENT_BRANCH" == "develop" ]] || [[ "$CURRENT_BRANCH" == release/* ]]; then
    BASE_BRANCH="$CURRENT_BRANCH"
else
    # Default to develop for non-base branches
    BASE_BRANCH="develop"
fi

print_info "Base branch: $BASE_BRANCH"

# Get the last commit that modified packages/ directory based on current state
# Always search from current branch or HEAD, never uses other branches
print_info "Searching for packages/ modification in current state"
PACKAGES_COMMIT=$(git log -1 --format=%H -- packages/ 2>/dev/null || true)

if [[ -z "$PACKAGES_COMMIT" ]]; then
    print_error "Cannot find any commit that modified packages/"
    exit 1
fi

# Get short commit hash (first 8 characters)
COMMIT_SHORT="${PACKAGES_COMMIT:0:8}"
print_info "Packages commit: $PACKAGES_COMMIT (short: $COMMIT_SHORT)"

# Get the commit date (when this commit was made), use this as the build date
DATE_STR=$(git log -1 --format=%cd --date=format:%Y%m%d "$PACKAGES_COMMIT")
print_info "Build date (from commit): $DATE_STR"

# Determine version suffix based on base branch
if [[ "$BASE_BRANCH" == release/* ]]; then
    VERSION_SUFFIX="post"
else
    VERSION_SUFFIX="dev"
fi

# Build the package version
PACKAGE_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${COMMIT_SHORT}"

print_info "Package version: $PACKAGE_VERSION"

# Build pip install command with extra index URL
BASE_URL="https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_SUFFIX}/"
EXTRA_INDEX_URL="--extra-index-url ${BASE_URL}"
print_info "Extra index URL: $BASE_URL"

# Install or download paddlefleet_ops using pip
if [[ -n "${PADDLEFLEET_OPS_DOWNLOAD_DIR:-}" ]]; then
    print_info "Downloading paddlefleet_ops ${PACKAGE_VERSION} to ${PADDLEFLEET_OPS_DOWNLOAD_DIR}..."
    mkdir -p "${PADDLEFLEET_OPS_DOWNLOAD_DIR}"
    if ! pip download --no-deps --dest "${PADDLEFLEET_OPS_DOWNLOAD_DIR}" "paddlefleet_ops==${PACKAGE_VERSION}" ${EXTRA_INDEX_URL}; then
        print_error "Failed to download paddlefleet_ops"
        print_info "The wheel may not be available yet. Please ensure the build has completed."
        print_info "You can manually check: ${BASE_URL}"
        exit 1
    fi
    print_info "Successfully downloaded paddlefleet_ops ${PACKAGE_VERSION}"
    exit 0
fi

print_info "Installing paddlefleet_ops ${PACKAGE_VERSION}..."
if ! pip install "paddlefleet_ops==${PACKAGE_VERSION}" ${EXTRA_INDEX_URL}; then
    print_error "Failed to install paddlefleet_ops"
    print_info "The wheel may not be available yet. Please ensure the build has completed."
    print_info "You can manually check: ${BASE_URL}"
    exit 1
fi

print_info "Successfully installed paddlefleet_ops ${PACKAGE_VERSION}"
