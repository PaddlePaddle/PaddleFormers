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
# Anchor resolution priority:
#   1. --commit <SHA>       → use the exact commit as anchor
#   2. --branch <name>      → use the branch HEAD as anchor
#   3. --from-setup [path]  → read locked commit or branch from PaddleFormers' setup.py
#   4. --from-env (default) → auto-detect from pip-installed paddlefleet:
#        - has commit hash (0.3.0.dev...+) → commit mode
#        - pure X.Y.Z (stable release)      → release/<major>.<minor> branch
#        - dev/post no hash / not installed  → develop branch
#
# If packages/ commits cannot be found via the anchor, falls back to develop branch.
# Uses GitHub API — no need to clone PaddleFleet locally.
#
# Usage:
#   ./install_ops_wheel.sh                            # auto from env (default)
#   ./install_ops_wheel.sh --branch develop
#   ./install_ops_wheel.sh --branch release/0.2
#   ./install_ops_wheel.sh --commit 30f17a82ef4
#   ./install_ops_wheel.sh --from-setup               # auto from ./setup.py
#   ./install_ops_wheel.sh --from-setup /path/to/PaddleFormers/setup.py

# NOT using set -e — all errors are handled explicitly for better diagnostics.
# If you see a failure, scroll up for the [ERROR] message.

# ============================================================
# Configuration
# ============================================================
PADDLE_FLEET_REPO="PaddlePaddle/PaddleFleet"
GITHUB_API="https://api.github.com/repos/${PADDLE_FLEET_REPO}"
GITHUB_RAW="https://raw.githubusercontent.com/${PADDLE_FLEET_REPO}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
print_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# Helpers
# ============================================================

# curl wrapper that returns the HTTP body (to stdout) and sets CURL_HTTP_CODE.
# Does NOT fail on non-200 — caller must check CURL_HTTP_CODE.
github_api_get() {
    local url="$1"
    local tmp
    tmp=$(mktemp)
    CURL_HTTP_CODE=$(curl -sL -w "%{http_code}" -o "$tmp" "$url" 2>/dev/null || echo "000")
    cat "$tmp"
    rm -f "$tmp"
}

# Pretty-print the GitHub API error summary when the response is not a list.
# Returns 0 if data looks valid (list), 1 if error.
check_api_response() {
    local json="$1"
    local label="$2"
    echo "$json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if isinstance(data, dict) and 'message' in data:
    print(f'[ERROR] GitHub API ({label}): {data[\"message\"]}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || return 1
    return 0
}

# Extract field(s) from JSON. Usage: get_json_field <json> <python-expr>
get_json_field() {
    local json="$1"
    local stmt="$2"
    echo "$json" | python3 -c "import json,sys; data=json.load(sys.stdin); $stmt" 2>/dev/null || echo ""
}

# Check if a branch exists in the remote repo
branch_exists() {
    local branch="$1"
    local http_code
    http_code=$(curl -sL -o /dev/null -w "%{http_code}" "${GITHUB_API}/branches/${branch}" 2>/dev/null || echo "000")
    [[ "$http_code" == "200" ]]
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

# Priority 1: --commit
if [[ -n "$FLEET_COMMIT" ]]; then
    MODE="commit"

# Priority 2: --branch
elif [[ -n "$FLEET_BRANCH" ]]; then
    MODE="branch"
    print_info "Mode: branch → $FLEET_BRANCH"

# Priority 3: --from-setup (read from PaddleFormers setup.py)
elif [[ -n "$FROM_SETUP" ]]; then
    SETUP_PY="$FROM_SETUP"
    [[ -d "$SETUP_PY" ]] && SETUP_PY="${SETUP_PY}/setup.py"
    if [[ ! -f "$SETUP_PY" ]]; then
        print_error "setup.py not found at $SETUP_PY"
        exit 1
    fi
    print_info "Reading PaddleFleet dependency from $SETUP_PY"

    # Try commit hash first: paddlefleet==X.Y.Z...<hash>
    FLEET_COMMIT=$(grep -oP 'paddlefleet==[0-9]+\.[0-9]+\.[0-9]+[^"]*\+\K[a-f0-9]{8,40}' "$SETUP_PY" 2>/dev/null || true)
    if [[ -n "$FLEET_COMMIT" ]]; then
        MODE="commit"
        print_info "Detected locked commit $FLEET_COMMIT from setup.py"
    else
        FLEET_BRANCH="develop"
        MODE="branch"
        print_info "No commit hash in setup.py, using branch: $FLEET_BRANCH"
    fi

# Priority 4: --from-env (default, auto-detect from pip-installed paddlefleet)
else
    MODE="env-auto"
    INSTALLED_VERSION=$(pip show paddlefleet 2>/dev/null | grep -oP '(?<=Version: )[0-9]+\.[0-9]+\.[0-9]+.*')

    if [[ -n "$INSTALLED_VERSION" ]]; then
        # Sub-case A: Version has embedded commit hash → commit mode
        FLEET_COMMIT=$(echo "$INSTALLED_VERSION" | grep -oP '\+\K[a-f0-9]{8,40}')
        if [[ -n "$FLEET_COMMIT" ]]; then
            MODE="commit"
            print_info "Found installed paddlefleet v${INSTALLED_VERSION} → commit $FLEET_COMMIT"
        else
            local_version=$(echo "$INSTALLED_VERSION" | grep -oP '^[0-9]+\.[0-9]+\.[0-9]+')
            # Sub-case B: Pure X.Y.Z (stable release) → release/<major>.<minor>
            if [[ "$INSTALLED_VERSION" == "$local_version" ]]; then
                RELEASE_BRANCH="release/$(echo "$local_version" | grep -oP '^[0-9]+\.[0-9]+')"
                print_info "Found installed paddlefleet v${INSTALLED_VERSION} (stable release)"
                if branch_exists "$RELEASE_BRANCH"; then
                    FLEET_BRANCH="$RELEASE_BRANCH"
                    print_info "→ mapped to branch: $FLEET_BRANCH"
                else
                    FLEET_BRANCH="develop"
                    print_warn "Branch $RELEASE_BRANCH not found, fallback to develop"
                fi
            # Sub-case C: dev/post version without commit hash → develop
            else
                FLEET_BRANCH="develop"
                print_info "Found installed paddlefleet v${INSTALLED_VERSION} (no hash) → develop"
            fi
        fi
    else
        FLEET_BRANCH="develop"
        print_info "paddlefleet not installed → develop"
    fi
fi

# ============================================================
# Step 2: Resolve anchor ref to a full commit SHA
# ============================================================
RESOLVED_SHA=""

if [[ "$MODE" == "commit" ]]; then
    print_info "Resolving commit: $FLEET_COMMIT"
    RESP=$(github_api_get "${GITHUB_API}/commits/${FLEET_COMMIT}")
    if [[ "$CURL_HTTP_CODE" != "200" ]]; then
        print_error "GitHub API returned HTTP $CURL_HTTP_CODE for commit $FLEET_COMMIT"
        print_error "Check the commit SHA or your network / API rate limit."
        exit 1
    fi
    check_api_response "$RESP" "commit $FLEET_COMMIT" || exit 1
    RESOLVED_SHA=$(get_json_field "$RESP" "print(data['sha'])")
    [[ -z "$RESOLVED_SHA" ]] && { print_error "Failed to parse commit SHA from API response"; exit 1; }
    print_info "Resolved: $RESOLVED_SHA"
else
    print_info "Resolving branch HEAD: $FLEET_BRANCH"
    RESP=$(github_api_get "${GITHUB_API}/branches/${FLEET_BRANCH}")
    if [[ "$CURL_HTTP_CODE" != "200" ]]; then
        print_error "GitHub API returned HTTP $CURL_HTTP_CODE for branch $FLEET_BRANCH"
        print_error "Check if the branch exists or your API rate limit."
        exit 1
    fi
    check_api_response "$RESP" "branch $FLEET_BRANCH" || exit 1
    RESOLVED_SHA=$(get_json_field "$RESP" "print(data['commit']['sha'])")
    [[ -z "$RESOLVED_SHA" ]] && { print_error "Failed to parse branch commit SHA"; exit 1; }
    print_info "Resolved ${FLEET_BRANCH} HEAD: $RESOLVED_SHA"
fi

# ============================================================
# Step 3: Fetch version.txt from the resolved commit
# ============================================================
print_info "Fetching version.txt at ${RESOLVED_SHA:0:12}..."
BASE_VERSION=$(curl -sL --fail "${GITHUB_RAW}/${RESOLVED_SHA}/version.txt" 2>/dev/null | head -1 | tr -d '[:space:]')
if [[ -z "$BASE_VERSION" ]]; then
    print_error "Failed to fetch version.txt at ${RESOLVED_SHA:0:12}"
    print_error "URL: ${GITHUB_RAW}/${RESOLVED_SHA}/version.txt"
    exit 1
fi
print_info "PaddleFleet version: $BASE_VERSION"

# ============================================================
# Step 4: Find the latest packages/ modification from this ref
# ============================================================
find_packages_commit() {
    local ref="$1"
    print_info "Searching packages/ last modification from ref ${ref:0:12}..."
    local RESP
    RESP=$(github_api_get "${GITHUB_API}/commits?path=packages/&sha=${ref}&per_page=1")
    if [[ "$CURL_HTTP_CODE" != "200" ]]; then
        print_warn "GitHub API returned HTTP $CURL_HTTP_CODE for packages/ history (rate limit?)"
        return 1
    fi
    check_api_response "$RESP" "packages/ history" || return 1

    local result
    result=$(get_json_field "$RESP" "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('NO_COMMIT')
    sys.exit(0)
c = data[0]
print(c['sha'], c['commit']['committer']['date'])
")
    if [[ -z "$result" || "$result" == "NO_COMMIT" ]]; then
        print_warn "No packages/ modification found from ref ${ref:0:12}"
        return 1
    fi
    echo "$result"
    return 0
}

PACKAGES_INFO=$(find_packages_commit "$RESOLVED_SHA") || {
    print_warn "Trying fallback: search packages/ from develop branch HEAD..."
    # Resolve develop branch HEAD
    RESP_DEV=$(github_api_get "${GITHUB_API}/branches/develop")
    if [[ "$CURL_HTTP_CODE" == "200" ]]; then
        DEV_SHA=$(get_json_field "$RESP_DEV" "print(data['commit']['sha'])")
        if [[ -n "$DEV_SHA" ]]; then
            print_info "Develop HEAD: $DEV_SHA"
            PACKAGES_INFO=$(find_packages_commit "$DEV_SHA") || {
                print_error "Failed to find packages/ commit from both anchor ref and develop branch."
                exit 1
            }
        else
            print_error "Failed to fallback to develop branch."
            exit 1
        fi
    else
        print_error "Failed to fallback to develop branch (HTTP $CURL_HTTP_CODE)."
        exit 1
    fi
}

PKG_SHA=$(echo "$PACKAGES_INFO" | awk '{print $1}')
PKG_DATE=$(echo "$PACKAGES_INFO" | awk '{print $2}')
PKG_SHORT="${PKG_SHA:0:8}"
DATE_STR=$(echo "$PKG_DATE" | tr -d '-' | cut -c1-8)

print_info "packages/ commit: $PKG_SHA"
print_info "Date: $PKG_DATE → $DATE_STR"

# ============================================================
# Step 5: Detect CUDA version
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
# Step 6: Build version & install
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