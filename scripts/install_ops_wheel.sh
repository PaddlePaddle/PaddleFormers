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

# Install paddlefleet_ops matching a given PaddleFleet ref.
#
# Two resolution strategies:
#   A. Version string with embedded packages/ commit hash (fast, no remote calls):
#        0.3.0.dev20260529+30f17a82ef4  → parse base/date/hash directly
#   B. Branch mode (no hash, need git):
#        develop / release/0.2          → git ls-remote + fetch to find packages/ commit
#
# Usage:
#   ./install_ops_wheel.sh                                # auto from env (default)
#   ./install_ops_wheel.sh --from-setup ./setup.py        # from PaddleFormers setup.py
#   ./install_ops_wheel.sh --commit 30f17a82ef4           # explicit commit (as fallback)
#   ./install_ops_wheel.sh --branch develop               # explicit branch
#   ./install_ops_wheel.sh --branch release/0.2

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
cleanup() { [[ -n "$GIT_DIR" && -d "$GIT_DIR" ]] && rm -rf "$GIT_DIR"; }
trap cleanup EXIT

git_init() {
    GIT_DIR=$(mktemp -d)
    git -C "$GIT_DIR" init -q --bare
}
git_run() {
    [[ -z "$GIT_DIR" ]] && git_init
    git -C "$GIT_DIR" "$@" 2>&1
}

# ============================================================
# Parse version string → extract version components
# Input:  "0.3.0.dev20260529+30f17a82ef4"
# Output: base="0.3.0" suffix="dev" date="20260529" hash="30f17a82ef4"
# ============================================================
parse_version() {
    local ver="$1"
    # Match: X.Y.Z.devDATE+HASH or X.Y.Z.postDATE+HASH
    if [[ "$ver" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.(dev|post)([0-9]{8})\+([a-f0-9]{8,40})$ ]]; then
        BASE_VERSION="${BASH_REMATCH[1]}"
        VERSION_SUFFIX="${BASH_REMATCH[2]}"
        DATE_STR="${BASH_REMATCH[3]}"
        OPS_HASH="${BASH_REMATCH[4]:0:8}"
        return 0
    fi
    return 1
}

# ============================================================
# Step 0: Parse arguments
# ============================================================
FLEET_COMMIT=""
FLEET_BRANCH=""
FROM_SETUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit)     FLEET_COMMIT="$2";      shift 2 ;;
        --branch)     FLEET_BRANCH="$2";      shift 2 ;;
        --from-setup)
            if [[ -n "$2" && "$2" != --* ]]; then
                FROM_SETUP="$2"; shift 2
            else
                FROM_SETUP="."; shift 1
            fi ;;
        --from-env)   shift 1 ;;
        *)
            print_error "Unknown: $1"
            echo "Usage: $0 [--commit <SHA> | --branch <name> | --from-setup [path] | --from-env]"
            exit 1 ;;
    esac
done

# ============================================================
# Step 1: Determine anchor — extract version info or fallback
# ============================================================
OPS_VERSION=""
NEED_GIT=false

# Priority 1: --commit (explicit hash, need git to find packages/)
if [[ -n "$FLEET_COMMIT" ]]; then
    NEED_GIT=true
    print_info "Mode: commit → $FLEET_COMMIT (using raw URLs + API)"
fi

# Priority 2: --branch
if [[ -n "$FLEET_BRANCH" ]]; then
    NEED_GIT=true
    print_info "Mode: branch → $FLEET_BRANCH"
fi

# Priority 3: --from-setup (read from setup.py)
if [[ -z "$FLEET_COMMIT" && -z "$FLEET_BRANCH" && -n "$FROM_SETUP" ]]; then
    SETUP_PY="$FROM_SETUP"
    [[ -d "$SETUP_PY" ]] && SETUP_PY="${SETUP_PY}/setup.py"
    if [[ ! -f "$SETUP_PY" ]]; then
        print_error "setup.py not found at $SETUP_PY"; exit 1
    fi

    # Extract the full version string: "paddlefleet==0.3.0.dev20260529+30f17a82ef4"
    FLEET_VER=$(grep -oP 'paddlefleet==\K[0-9]+\.[0-9]+\.[0-9]+[^"]*' "$SETUP_PY" 2>/dev/null || true)
    if [[ -n "$FLEET_VER" ]] && parse_version "$FLEET_VER"; then
        OPS_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${OPS_HASH}"
        print_info "Parsed from setup.py: base=$BASE_VERSION, date=$DATE_STR, hash=$OPS_HASH"
        print_info "→ paddlefleet_ops==${OPS_VERSION}"
    else
        print_warn "No commit hash in setup.py version (${FLEET_VER:-none}), fallback to develop"
        NEED_GIT=true
        FLEET_BRANCH="develop"
    fi
fi

# Priority 4: --from-env (default, auto-detect from pip-installed paddlefleet)
if [[ -z "$FLEET_COMMIT" && -z "$FLEET_BRANCH" && -z "$OPS_VERSION" ]]; then
    INSTALLED_VER=$(pip show paddlefleet 2>/dev/null | grep -oP '(?<=Version: )[0-9]+\.[0-9]+\.[0-9]+.*')
    if [[ -n "$INSTALLED_VER" ]] && parse_version "$INSTALLED_VER"; then
        OPS_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${OPS_HASH}"
        print_info "Parsed from installed paddlefleet v${INSTALLED_VER}"
        print_info "→ paddlefleet_ops==${OPS_VERSION}"
    elif [[ -n "$INSTALLED_VER" ]]; then
        # Pure version without hash (e.g. "0.3.0" or "0.3.0.dev20260601")
        local_version=$(echo "$INSTALLED_VER" | grep -oP '^[0-9]+\.[0-9]+\.[0-9]+')
        if [[ "$INSTALLED_VER" == "$local_version" ]]; then
            FLEET_BRANCH="release/$(echo "$local_version" | grep -oP '^[0-9]+\.[0-9]+')"
            print_info "Installed paddlefleet v${INSTALLED_VER} (stable), trying branch: $FLEET_BRANCH"
        else
            FLEET_BRANCH="develop"
            print_info "Installed paddlefleet v${INSTALLED_VER} (no hash), using branch: $FLEET_BRANCH"
        fi
        NEED_GIT=true
    else
        FLEET_BRANCH="develop"
        NEED_GIT=true
        print_info "paddlefleet not installed, using branch: develop"
    fi
fi

# ============================================================
# Step 2: If version was parsed directly, skip git path
# ============================================================
if [[ -n "$OPS_VERSION" ]]; then
    print_info "All version info available locally, skipping git fetch."
fi

# ============================================================
# Step 3 (git path): Resolve branch → find packages/ commit
# Only reached when NEED_GIT=true
# ============================================================
BASE_VERSION=""
DATE_STR=""
OPS_HASH=""

if $NEED_GIT; then
    if [[ -n "$FLEET_COMMIT" ]]; then
        # --commit mode: no branch, use raw URLs + commit itself
        print_info "Commit mode: fetching info for $FLEET_COMMIT via raw URLs..."
        BASE_VERSION=$(curl -sL --fail "${GITHUB_RAW}/${FLEET_COMMIT}/version.txt" 2>/dev/null | head -1 | tr -d '[:space:]') || {
            print_error "Cannot fetch version.txt at commit $FLEET_COMMIT"
            print_error "URL: ${GITHUB_RAW}/${FLEET_COMMIT}/version.txt"
            exit 1
        }
        OPS_HASH="${FLEET_COMMIT:0:8}"
        VERSION_SUFFIX="dev"
        # Use commit date (best-effort, fallback to today if unavailable)
        DATE_STR=$(date +%Y%m%d)
        print_warn "Using today's date for commit mode: $DATE_STR (commit date unavailable via raw URLs)"
        OPS_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${OPS_HASH}"
        print_info "Commit resolved: base=$BASE_VERSION, date=$DATE_STR, hash=$OPS_HASH"
        print_info "→ paddlefleet_ops==${OPS_VERSION}"

    else
        # 3a. Check branch exists
        BRANCH_SHA=$(git_run ls-remote "$PADDLE_FLEET_URL" "refs/heads/${FLEET_BRANCH}" 2>/dev/null | awk '{print $1}')
        if [[ -z "$BRANCH_SHA" ]]; then
            print_warn "Branch '$FLEET_BRANCH' not found, fallback to develop"
            FLEET_BRANCH="develop"
            BRANCH_SHA=$(git_run ls-remote "$PADDLE_FLEET_URL" "refs/heads/develop" | awk '{print $1}')
            if [[ -z "$BRANCH_SHA" ]]; then
                print_error "Cannot reach github.com. Check your network."
                exit 1
            fi
        fi

        # 3b. Fetch the branch
        print_info "Fetching branch: $FLEET_BRANCH (${BRANCH_SHA:0:12})"
        git_run fetch --depth=200 --no-tags "$PADDLE_FLEET_URL" "$FLEET_BRANCH" 2>/dev/null || {
            print_error "Git fetch failed for $FLEET_BRANCH. Network issue?"
            exit 1
        }

        # 3c. Get version.txt
        BASE_VERSION=$(git_run show "FETCH_HEAD:version.txt" 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ -z "$BASE_VERSION" ]]; then
            print_warn "version.txt via git failed, trying raw URL..."
            BASE_VERSION=$(curl -sL --fail "${GITHUB_RAW}/${BRANCH_SHA}/version.txt" 2>/dev/null | head -1 | tr -d '[:space:]') || true
        fi
        if [[ -z "$BASE_VERSION" ]]; then
            print_error "Cannot fetch version.txt. Try specifying a commit with --commit <SHA>"
            exit 1
        fi

        # 3d. Find packages/ commit via git log
        PACKAGES_INFO=$(git_run log "FETCH_HEAD" -1 --format="%H %cd" --date=format:"%Y%m%d" -- "packages/" 2>/dev/null) || true

        if [[ -z "$PACKAGES_INFO" ]]; then
            # Try deeper history
            print_warn "Shallow log too short, trying depth=500..."
            git_run fetch --depth=500 --no-tags "$PADDLE_FLEET_URL" "$FLEET_BRANCH" 2>/dev/null || true
            PACKAGES_INFO=$(git_run log "FETCH_HEAD" -1 --format="%H %cd" --date=format:"%Y%m%d" -- "packages/" 2>/dev/null) || true
        fi

        if [[ -z "$PACKAGES_INFO" ]]; then
            # Fallback: use FETCH_HEAD itself as packages/ commit
            PKG_SHA="$BRANCH_SHA"
            PKG_SHORT="${PKG_SHA:0:8}"
            DATE_STR=$(date +%Y%m%d)
            print_warn "Could not find packages/ commit history. Using branch HEAD as fallback."
            print_warn "  → commit: $PKG_SHORT, date: $DATE_STR (today)"
        else
            PKG_SHA=$(echo "$PACKAGES_INFO" | awk '{print $1}')
            DATE_STR=$(echo "$PACKAGES_INFO" | awk '{print $2}')
            PKG_SHORT="${PKG_SHA:0:8}"
        fi

        # Determine version suffix
        VERSION_SUFFIX="dev"
        [[ "$FLEET_BRANCH" == release/* ]] && VERSION_SUFFIX="post"

        OPS_HASH="$PKG_SHORT"
        OPS_VERSION="${BASE_VERSION}.${VERSION_SUFFIX}${DATE_STR}+${OPS_HASH}"
        print_info "Branch resolved: base=$BASE_VERSION, suffix=$VERSION_SUFFIX, date=$DATE_STR, hash=$OPS_HASH"
        print_info "→ paddlefleet_ops==${OPS_VERSION}"
    fi  # end commit vs branch
fi  # end NEED_GIT

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
# Step 5: Install
# ============================================================
WHEEL_URL="https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_SUFFIX}/"

print_info "Installing paddlefleet_ops==${OPS_VERSION}"
print_info "Index: $WHEEL_URL"

if pip install "paddlefleet_ops==${OPS_VERSION}" --extra-index-url "${WHEEL_URL}"; then
    print_info "Successfully installed paddlefleet_ops ${OPS_VERSION}"
else
    print_error "pip install failed for paddlefleet_ops==${OPS_VERSION}"
    print_error "Check if the wheel exists at: ${WHEEL_URL}"
    exit 1
fi