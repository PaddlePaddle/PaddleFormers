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

# Script to download and install paddleformers_ops wheel based on local git state.
# Automatically detects the current CUDA version. Only CUDA 13.2, CUDA 13.0 and CUDA 12.9 are supported.

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
    elif [ -n "$CUDA_HOME" ]; then
        cuda_ver=$(basename "$CUDA_HOME" | grep -oP '[0-9]+\.[0-9]+' || echo "")
    elif [ -n "$CUDA_PATH" ]; then
        cuda_ver=$(basename "$CUDA_PATH" | grep -oP '[0-9]+\.[0-9]+' || echo "")
    fi
    
    echo "$cuda_ver"
}

# Get CUDA suffix for package naming
get_cuda_suffix() {
    local cuda_ver="$1"
    
    case "$cuda_ver" in
        13.2)
            echo "cu132"
            ;;
        13.0)
            echo "cu130"
            ;;
        12.9)
            echo "cu129"
            ;;
        *)
            return 1
            ;;
    esac
}

# Main script
print_info "Detecting CUDA version..."

DETECTED_CUDA_VERSION=$(detect_cuda_version)

if [ -z "$DETECTED_CUDA_VERSION" ]; then
    print_error "Could not detect CUDA version. Please ensure CUDA is installed and accessible."
    print_error "Supported versions: CUDA 13.2, CUDA 13.0, CUDA 12.9"
    exit 1
fi

CUDA_SUFFIX=$(get_cuda_suffix "$DETECTED_CUDA_VERSION")
if [ $? -ne 0 ]; then
    print_error "Only CUDA 13.2, CUDA 13.0 and CUDA 12.9 are supported."
    print_error "Detected version: $DETECTED_CUDA_VERSION"
    exit 1
fi

print_info "Using CUDA version: $DETECTED_CUDA_VERSION (suffix: $CUDA_SUFFIX)"

# Get workspace root (assuming this script is run from the repository root)
WORKSPACE_ROOT="$(git rev-parse --show-toplevel)"
cd "$WORKSPACE_ROOT"

# Get base version from setup.py
# Version format: "1.1.0.dev" -> extract "1.1.0"
BASE_VERSION=$(grep -oP '__version__\s*=\s*"\K[0-9]+\.[0-9]+\.[0-9]+' setup.py | head -1)

if [ -z "$BASE_VERSION" ]; then
    print_error "Could not extract base version from setup.py"
    exit 1
fi

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

# Get the last commit that modified ops/ or csrc/ directory
# Adjust this path based on where your ops source code is located
print_info "Searching for ops modification in current state"

# Try multiple possible paths for ops source code
OPS_COMMIT=""
for ops_path in "ops" "csrc" "paddleformers/ops" "paddleformers/csrc"; do
    OPS_COMMIT=$(git log -1 --format=%H -- "$ops_path" 2>/dev/null || true)
    if [ -n "$OPS_COMMIT" ]; then
        print_info "Found ops commit from path: $ops_path"
        break
    fi
done

if [ -z "$OPS_COMMIT" ]; then
    print_warn "Cannot find any commit that modified ops directory"
    print_info "Using HEAD commit instead"
    OPS_COMMIT=$(git rev-parse HEAD)
fi

# Get short commit hash (first 8 characters)
COMMIT_SHORT="${OPS_COMMIT:0:8}"

print_info "Ops commit: $OPS_COMMIT (short: $COMMIT_SHORT)"

# Get the commit date (when this commit was made), use this as the build date
DATE_STR=$(git log -1 --format=%cd --date=format:%Y%m%d "$OPS_COMMIT")
print_info "Build date (from commit): $DATE_STR"

# Determine version suffix based on base branch
if [[ "$BASE_BRANCH" == "develop" ]]; then
    VERSION_SUFFIX="dev"
else
    VERSION_SUFFIX="post"
fi

# Build the package version
PACKAGE_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${COMMIT_SHORT}"

print_info "Package version: $PACKAGE_VERSION"

# Build pip install command with extra index URL
# TODO: Update this URL to the actual wheel storage location
BASE_URL="https://paddle-qa.bj.bcebos.com/paddleformers/ops/${CUDA_SUFFIX}"
EXTRA_INDEX_URL="--extra-index-url ${BASE_URL}"

print_info "Extra index URL: $BASE_URL"

# Install paddleformers_ops using pip
print_info "Installing paddleformers_ops ${PACKAGE_VERSION}..."

if ! pip install "paddleformers_ops==${PACKAGE_VERSION}" ${EXTRA_INDEX_URL}; then
    print_error "Failed to install paddleformers_ops"
    print_info "The wheel may not be available yet. Please ensure the build has completed."
    print_info "You can manually check: ${BASE_URL}"
    exit 1
fi

print_info "Successfully installed paddleformers_ops ${PACKAGE_VERSION}"
