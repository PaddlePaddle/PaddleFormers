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

# Script to install paddlefleet_ops that matches a given PaddleFleet ref.
#
# Uses git protocol (not GitHub REST API) to avoid API rate limits in
# restricted network environments. All remote operations are shallow
# fetches — no need to clone PaddleFleet fully.
#
# Anchor resolution priority:
#   1. --commit <SHA>       → use the exact commit as anchor
#   2. --branch <name>      → use the branch HEAD as anchor
#   3. --from-setup [path]  → read locked commit from PaddleFormers' setup.py
#   4. --from-env (default) → auto-detect from pip-installed paddlefleet
#
# If packages/ commits cannot be found via the anchor, falls back to develop.
#
# Usage:
#   ./install_ops_wheel.sh                            # auto from env (default)
#   ./install_ops_wheel.sh --branch develop
#   ./install_ops_wheel.sh --branch release/0.2
#   ./install_ops_wheel.sh --commit 30f17a82ef4
#   ./install_ops_wheel.sh --from-setup ./setup.py

set -u

# ============================================================
# Configuration
# ============================================================
PADDLE_FLEET_URL="https://github.com/PaddlePaddle/PaddleFleet.git"
PADDLE_FLEET_REPO="PaddlePaddle/PaddleFleet"
GITHUB_RAW="https://raw.githubusercontent.com/${PADDLE_FLEET_REPO}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
print_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

GIT_DIR=""

cleanup() {
    if [[ -n "$GIT_DIR" && -d "$GIT_DIR" ]]; then
        rm -rf "$GIT_DIR"
    fi
}
trap cleanup EXIT

# ============================================================
# Helpers
# ============================================================

# git_run <args...> — run git in a temporary bare repo
git_run() {
    if [[ -z "$GIT_DIR" ]]; then
        GIT_DIR=$(mktemp -d)
        git -C "$GIT_DIR" init -q --bare
    fi
    git -C "$GIT_DIR" "$@" 2>&1
}

# Fetch enough history from a given ref to walk the commit graph.
# Uses --shallow-since to get a wide enough window.
git_fetch_ref() {
    local ref="$1"
    local since="${2:-2025-01-01}"
    print_info "Fetching git history for $ref (since $since)..."
    local output
    output=$(git_run fetch --shallow-since="$since" --no-tags "$PADDLE_FLEET_URL" "$ref" 2>&1) || {
        print_warn "Shallow fetch failed (network issue?), retrying with depth=200..."
        output=$(git_run fetch --depth=200 --no-tags "$PADDLE_FLEET_URL" "$ref" 2>&1) || {
            print_error "Git fetch failed. Check your network."
            print_error "Command: git fetch --depth=200 --no-tags $PADDLE_FLEET_URL $ref"
            print_error "Output: $(echo "$output" | tail -3)"
            return 1
        }
    }
    return 0
}

# ============================================================
# Step 0: Parse arguments
# ============================================================
FLEET_COMMIT=""
FLEET_BRANCH=""
FROM_SETUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit)
            FLEET_COMMIT="$2"
            shift 2
            ;;
        --branch)
            FLEET_BRANCH="$2"
            shift 2
            ;;
        --from-setup)
            if [[ -n "$2" && "$2" != --* ]]; then
                FROM_SETUP="$2"
                shift 2
            else
                FROM_SETUP="."
                shift 1
            fi
            ;;
        --from-env)
            shift 1
            ;;
        *)
            print_error "Unknown argument: $1"
            echo "Usage: $0 [--commit <SHA> | --branch <name> | --from-setup [path] | --from-env]"
            exit 1
            ;;
    esac
done

# ============================================================
# Step 1: Determine anchor ref
# ============================================================
MODE=""

if [[ -n "$FLEET_COMMIT" ]]; then
    MODE="commit"

elif [[ -n "$FLEET_BRANCH" ]]; then
    MODE="branch"
    print_info "Mode: branch → $FLEET_BRANCH"

elif [[ -n "$FROM_SETUP" ]]; then
    SETUP_PY="$FROM_SETUP"
    [[ -d "$SETUP_PY" ]] && SETUP_PY="${SETUP_PY}/setup.py"
    if [[ ! -f "$SETUP_PY" ]]; then
        print_error "setup.py not found at $SETUP_PY"
        exit 1
    fi
    print_info "Reading PaddleFleet dependency from $SETUP_PY"

    FLEET_COMMIT=$(grep -oP 'paddlefleet==[0-9]+\.[0-9]+\.[0-9]+[^"]*\+\K[a-f0-9]{8,40}' "$SETUP_PY" 2>/dev/null || true)
    if [[ -n "$FLEET_COMMIT" ]]; then
        MODE="commit"
        print_info "Detected locked commit $FLEET_COMMIT from setup.py"
    else
        FLEET_BRANCH="develop"
        MODE="branch"
        print_info "No commit hash in setup.py, using branch: $FLEET_BRANCH"
    fi

else
    MODE="env-auto"
    INSTALLED_VERSION=$(pip show paddlefleet 2>/dev/null | grep -oP '(?<=Version: )[0-9]+\.[0-9]+\.[0-9]+.*')

    if [[ -n "$INSTALLED_VERSION" ]]; then
        FLEET_COMMIT=$(echo "$INSTALLED_VERSION" | grep -oP '\+\K[a-f0-9]{8,40}')
        if [[ -n "$FLEET_COMMIT" ]]; then
            MODE="commit"
            print_info "Installed paddlefleet v${INSTALLED_VERSION} → commit $FLEET_COMMIT"
        else
            local_version=$(echo "$INSTALLED_VERSION" | grep -oP '^[0-9]+\.[0-9]+\.[0-9]+')
            if [[ "$INSTALLED_VERSION" == "$local_version" ]]; then
                RELEASE_BRANCH="release/$(echo "$local_version" | grep -oP '^[0-9]+\.[0-9]+')"
                print_info "Installed paddlefleet v${INSTALLED_VERSION} (stable release)"
                # Check branch existence via git ls-remote
                BRANCH_EXISTS=$(git_run ls-remote --heads "$PADDLE_FLEET_URL" "$RELEASE_BRANCH" 2>/dev/null | wc -l)
                if [[ "$BRANCH_EXISTS" -gt 0 ]]; then
                    FLEET_BRANCH="$RELEASE_BRANCH"
                    print_info "→ mapped to branch: $FLEET_BRANCH"
                else
                    FLEET_BRANCH="develop"
                    print_warn "Branch $RELEASE_BRANCH not found, fallback to develop"
                fi
            else
                FLEET_BRANCH="develop"
                print_info "Installed paddlefleet v${INSTALLED_VERSION} (no hash) → develop"
            fi
        fi
    else
        FLEET_BRANCH="develop"
        print_info "paddlefleet not installed → develop"
    fi
fi

# ============================================================
# Step 2: Resolve anchor ref & get version.txt
# ============================================================
RESOLVED_SHA=""
BASE_VERSION=""
FETCH_REF=""

if [[ "$MODE" == "commit" ]]; then
    # Fetch just enough to resolve the commit and get version.txt
    FETCH_REF="$FLEET_COMMIT"
    git_fetch_ref "$FETCH_REF" || exit 1

    RESOLVED_SHA=$(git_run rev-parse "FETCH_HEAD" 2>/dev/null) || {
        print_error "Failed to resolve commit $FLEET_COMMIT after fetch"
        exit 1
    }
    print_info "Resolved commit: $RESOLVED_SHA"

    # Get version.txt from that commit
    BASE_VERSION=$(git_run show "FETCH_HEAD:version.txt" 2>/dev/null | head -1 | tr -d '[:space:]') || true
else
    # Branch mode: get branch HEAD via ls-remote, then fetch
    RESOLVED_SHA=$(git_run ls-remote "$PADDLE_FLEET_URL" "refs/heads/${FLEET_BRANCH}" 2>/dev/null | awk '{print $1}')
    if [[ -z "$RESOLVED_SHA" ]]; then
        print_error "Branch '$FLEET_BRANCH' not found in PaddleFleet remote"
        exit 1
    fi
    print_info "Resolved ${FLEET_BRANCH} HEAD: $RESOLVED_SHA"

    FETCH_REF="$RESOLVED_SHA"
    git_fetch_ref "$FETCH_REF" || exit 1

    BASE_VERSION=$(git_run show "FETCH_HEAD:version.txt" 2>/dev/null | head -1 | tr -d '[:space:]') || true
fi

if [[ -z "$BASE_VERSION" ]]; then
    # Fallback: try via raw.githubusercontent.com
    print_warn "version.txt not found via git, trying raw URL..."
    BASE_VERSION=$(curl -sL --fail "${GITHUB_RAW}/${RESOLVED_SHA}/version.txt" 2>/dev/null | head -1 | tr -d '[:space:]') || true
fi
if [[ -z "$BASE_VERSION" ]]; then
    print_error "Failed to fetch version.txt"
    exit 1
fi
print_info "PaddleFleet version: $BASE_VERSION"

# ============================================================
# Step 3: Find the latest packages/ modification from this ref
# ============================================================
find_packages_commit_from_ref() {
    local ref="$1"
    # git log -- packages/ walks the commit graph from the given ref
    # This works if we have sufficient history
    local result
    result=$(git_run log "FETCH_HEAD" -1 --format="%H %cd" --date=format:"%Y%m%d" -- "packages/" 2>/dev/null) || true
    echo "$result"
}

PACKAGES_INFO=$(find_packages_commit_from_ref "$FETCH_REF")

if [[ -z "$PACKAGES_INFO" ]]; then
    print_warn "packages/ not found from anchor ref, trying raw URL fallback..."

    # Fallback: use the commit itself (the packages/ dir may be in this commit)
    # Try raw.githubusercontent.com to check if packages/ exists at this commit
    PKG_TEST=$(curl -sL --fail "${GITHUB_RAW}/${RESOLVED_SHA}/packages/" 2>/dev/null || true)
    if [[ -n "$PKG_TEST" ]]; then
        # The commit itself contains packages/ changes, use the commit directly
        PKG_SHA="$RESOLVED_SHA"
        PKG_SHORT="${PKG_SHA:0:8}"
        # Use raw URL to get commit date
        DATE_STR=$(curl -s "${GITHUB_API:-https://api.github.com}/repos/${PADDLE_FLEET_REPO}/commits/${RESOLVED_SHA}" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['commit']['committer']['date'][:10].replace('-',''))" 2>/dev/null || echo "")
        if [[ -z "$DATE_STR" ]]; then
            DATE_STR=$(date +%Y%m%d)
            print_warn "Could not determine date, using today: $DATE_STR"
        fi
        print_info "Using anchor commit directly for packages/: $PKG_SHA (date: $DATE_STR)"
    else
        print_warn "Trying fallback: develop branch..."
        # Resolve develop HEAD via ls-remote
        DEV_SHA=$(git_run ls-remote "$PADDLE_FLEET_URL" "refs/heads/develop" 2>/dev/null | awk '{print $1}')
        if [[ -z "$DEV_SHA" ]]; then
            print_error "Failed to resolve develop branch"
            exit 1
        fi
        print_info "Develop HEAD: $DEV_SHA"
        git_fetch_ref "$DEV_SHA" || exit 1

        PACKAGES_INFO=$(git_run log "FETCH_HEAD" -1 --format="%H %cd" --date=format:"%Y%m%d" -- "packages/" 2>/dev/null) || true
        if [[ -z "$PACKAGES_INFO" ]]; then
            # One more fallback: deeper history
            print_warn "Still no packages/ commit found, trying depth=500..."
            git_run fetch --depth=500 --no-tags "$PADDLE_FLEET_URL" "$DEV_SHA" 2>/dev/null || true
            PACKAGES_INFO=$(git_run log "FETCH_HEAD" -1 --format="%H %cd" --date=format:"%Y%m%d" -- "packages/" 2>/dev/null) || true
        fi
    fi
fi

if [[ -z "$PACKAGES_INFO" ]]; then
    print_error "Cannot find packages/ commit from any source."
    exit 1
fi

if [[ -z "$PKG_SHA" ]]; then
    # Parse from git log output
    PKG_SHA=$(echo "$PACKAGES_INFO" | awk '{print $1}')
    DATE_STR=$(echo "$PACKAGES_INFO" | awk '{print $2}')
    PKG_SHORT="${PKG_SHA:0:8}"
fi

print_info "packages/ commit: $PKG_SHA (short: $PKG_SHORT, date: $DATE_STR)"

# ============================================================
# Step 4: Detect CUDA version
# ============================================================
detect_cuda_version() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+'
    elif [[ -n "${CUDA_HOME:-}" ]]; then
        "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'
    elif [[ -n "${CUDA_PATH:-}" ]]; then
        "${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'
    fi
}

CUDA_VER=$(detect_cuda_version)
if [[ -z "$CUDA_VER" ]]; then
    print_error "Cannot detect CUDA version. Install CUDA toolkit or ensure nvidia-smi works."
    exit 1
fi
print_info "CUDA: $CUDA_VER"

case "$CUDA_VER" in
    "13.2") CUDA_SUFFIX="cu132" ;;
    "13.0") CUDA_SUFFIX="cu130" ;;
    "12.9") CUDA_SUFFIX="cu129" ;;
    *)
        print_error "Unsupported CUDA: $CUDA_VER (supported: 13.2, 13.0, 12.9)"
        exit 1
        ;;
esac
print_info "CUDA suffix: $CUDA_SUFFIX"

# ============================================================
# Step 5: Build version & install
# ============================================================
VERSION_SUFFIX="dev"
if [[ -n "$FLEET_BRANCH" && "$FLEET_BRANCH" == release/* ]]; then
    VERSION_SUFFIX="post"
fi

PKG_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${PKG_SHORT}"
WHEEL_URL="https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_SUFFIX}/"

print_info "Target: paddlefleet_ops==${PKG_VERSION}"
print_info "Index:  $WHEEL_URL"

if pip install "paddlefleet_ops==${PKG_VERSION}" --extra-index-url "${WHEEL_URL}"; then
    print_info "Successfully installed paddlefleet_ops ${PKG_VERSION}"
else
    print_error "pip install failed for paddlefleet_ops==${PKG_VERSION}"
    print_error "The wheel may not be available yet. Check: ${WHEEL_URL}"
    exit 1
fi