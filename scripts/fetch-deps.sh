#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DOWNLOAD_DIR="$PROJECT_ROOT/deps/.work/downloads"
CHECK_ONLY=false

usage() {
    cat <<'EOF'
Usage: scripts/fetch-deps.sh [--check] {all|toolchain|spike|model|tflite-reference}

Without --check, fetch and install the dependency pinned by its TOML manifest.
With --check, validate the local dependency without accessing the network.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

# Read the string values used by this project's small, checked-in TOML
# manifests. Keeping bootstrap code dependency-free lets this run before the
# uv environments have been created.
toml_string() {
    local file=$1 section=$2 key=$3 value
    value=$(
        awk -v wanted_section="[$section]" -v wanted_key="$key" '
            /^[[:space:]]*\[/ {
                active = ($0 == wanted_section)
                next
            }
            active {
                line = $0
                sub(/^[[:space:]]*/, "", line)
                if (index(line, wanted_key) != 1) next
                rest = substr(line, length(wanted_key) + 1)
                if (rest !~ /^[[:space:]]*=/) next
                sub(/^[[:space:]]*=[[:space:]]*/, "", rest)
                if (rest !~ /^"[^"]*"[[:space:]]*$/) next
                sub(/^"/, "", rest)
                sub(/"[[:space:]]*$/, "", rest)
                print rest
                exit
            }
        ' "$file"
    )
    [[ -n "$value" ]] || die "missing [$section].$key in ${file#$PROJECT_ROOT/}"
    printf '%s\n' "$value"
}

resolve_project_path() {
    local relative=$1
    [[ "$relative" != /* ]] || die "manifest path must be relative: $relative"
    [[ "/$relative/" != *"/../"* ]] || die "manifest path may not contain '..': $relative"
    printf '%s/%s\n' "$PROJECT_ROOT" "$relative"
}

verify_sha256() {
    local file=$1 expected=$2 actual
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA256 in manifest: $expected"
    [[ -f "$file" ]] || return 1
    actual=$(sha256sum "$file" | awk '{print $1}')
    [[ "$actual" == "$expected" ]]
}

download() {
    local url=$1 expected=$2 output=$3 temporary
    mkdir -p "$(dirname -- "$output")"

    if verify_sha256 "$output" "$expected"; then
        printf 'Using cached %s\n' "${output#$PROJECT_ROOT/}"
        return
    fi

    temporary="${output}.part"
    rm -f -- "$temporary"
    printf 'Downloading %s\n' "$url"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "$temporary" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --output-document="$temporary" "$url"
    else
        die "curl or wget is required"
    fi

    verify_sha256 "$temporary" "$expected" || {
        rm -f -- "$temporary"
        die "SHA256 mismatch for $(basename -- "$output")"
    }
    mv -f -- "$temporary" "$output"
}

check_toolchain() {
    local manifest="$PROJECT_ROOT/deps/manifests/riscv-toolchain.toml"
    local prefix gcc_version gcc
    prefix=$(resolve_project_path "$(toml_string "$manifest" toolchain install_prefix)")
    gcc_version=$(toml_string "$manifest" toolchain gcc_version)
    gcc="$prefix/bin/riscv64-unknown-elf-gcc"
    [[ -x "$gcc" ]] || die "toolchain is missing; run 'make fetch-toolchain'"
    "$gcc" --version | head -n 1 | grep -Fq "$gcc_version" ||
        die "installed toolchain does not report GCC $gcc_version"
    printf 'OK toolchain: GCC %s\n' "$gcc_version"
}

fetch_toolchain() {
    local manifest="$PROJECT_ROOT/deps/manifests/riscv-toolchain.toml"
    local archive url expected prefix cached staging payload
    archive=$(toml_string "$manifest" toolchain archive)
    url=$(toml_string "$manifest" toolchain url)
    expected=$(toml_string "$manifest" toolchain sha256)
    prefix=$(resolve_project_path "$(toml_string "$manifest" toolchain install_prefix)")
    cached="$DOWNLOAD_DIR/$archive"

    if [[ -x "$prefix/bin/riscv64-unknown-elf-gcc" ]] &&
       "$prefix/bin/riscv64-unknown-elf-gcc" --version | head -n 1 | grep -Fq "$(toml_string "$manifest" toolchain gcc_version)"; then
        printf 'OK toolchain: GCC %s\n' "$(toml_string "$manifest" toolchain gcc_version)"
        return
    fi

    download "$url" "$expected" "$cached"
    [[ ! -e "$prefix" ]] || die "$prefix already exists but does not match the manifest; move it aside and retry"

    mkdir -p "$PROJECT_ROOT/deps/.work" "$(dirname -- "$prefix")"
    staging=$(mktemp -d "$PROJECT_ROOT/deps/.work/toolchain.XXXXXX")
    trap 'rm -rf -- "$staging"' RETURN
    tar -xJf "$cached" -C "$staging"

    payload="$staging"
    if [[ -d "$staging/riscv" ]]; then
        payload="$staging/riscv"
    fi
    mv -- "$payload" "$prefix"
    trap - RETURN
    [[ "$payload" == "$staging" ]] || rm -rf -- "$staging"
    check_toolchain
}

check_spike() {
    local manifest="$PROJECT_ROOT/deps/manifests/spike.toml"
    local source expected actual
    source=$(resolve_project_path "$(toml_string "$manifest" spike source)")
    expected=$(toml_string "$manifest" spike commit)
    [[ -d "$source/.git" ]] || die "Spike source is missing; run 'make fetch-spike'"
    actual=$(git -C "$source" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] || die "Spike is at $actual, expected $expected"
    [[ -z "$(git -C "$source" status --short)" ]] || die "Spike source has uncommitted changes"
    printf 'OK spike: %s\n' "$expected"
}

fetch_spike() {
    local manifest="$PROJECT_ROOT/deps/manifests/spike.toml"
    local repository commit source origin
    repository=$(toml_string "$manifest" spike repository)
    commit=$(toml_string "$manifest" spike commit)
    source=$(resolve_project_path "$(toml_string "$manifest" spike source)")

    if [[ -d "$source/.git" ]]; then
        origin=$(git -C "$source" remote get-url origin)
        [[ "$origin" == "$repository" ]] || die "Spike origin is $origin, expected $repository"
        if [[ "$(git -C "$source" rev-parse HEAD)" == "$commit" ]]; then
            check_spike
            return
        fi
        [[ -z "$(git -C "$source" status --short)" ]] || die "refusing to replace modified Spike source"
    else
        [[ ! -e "$source" ]] || rmdir "$source" 2>/dev/null || die "$source exists and is not an empty Git checkout"
        mkdir -p "$source"
        git -C "$source" init --quiet
        git -C "$source" remote add origin "$repository"
    fi

    git -C "$source" fetch --depth 1 origin "$commit"
    git -C "$source" checkout --quiet --detach FETCH_HEAD
    check_spike
}

check_git_source() {
    local section=$1 label=$2
    local manifest="$PROJECT_ROOT/deps/manifests/tflite-reference.toml"
    local source expected actual
    source=$(resolve_project_path "$(toml_string "$manifest" "$section" source)")
    expected=$(toml_string "$manifest" "$section" commit)
    [[ -d "$source/.git" ]] || die "$label source is missing; run 'make fetch-tflite-reference'"
    actual=$(git -C "$source" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] || die "$label is at $actual, expected $expected"
    [[ -z "$(git -C "$source" status --short)" ]] || die "$label source has uncommitted changes"
    printf 'OK %s: %s\n' "$label" "$expected"
}

fetch_git_source() {
    local section=$1 label=$2 sparse_path=${3:-}
    local manifest="$PROJECT_ROOT/deps/manifests/tflite-reference.toml"
    local repository commit source origin
    repository=$(toml_string "$manifest" "$section" repository)
    commit=$(toml_string "$manifest" "$section" commit)
    source=$(resolve_project_path "$(toml_string "$manifest" "$section" source)")

    if [[ -d "$source/.git" ]]; then
        origin=$(git -C "$source" remote get-url origin)
        [[ "$origin" == "$repository" ]] || die "$label origin is $origin, expected $repository"
        if [[ "$(git -C "$source" rev-parse HEAD)" == "$commit" ]]; then
            check_git_source "$section" "$label"
            return
        fi
        [[ -z "$(git -C "$source" status --short)" ]] || die "refusing to replace modified $label source"
    else
        [[ ! -e "$source" ]] || rmdir "$source" 2>/dev/null || die "$source exists and is not an empty Git checkout"
        mkdir -p "$source"
        git -C "$source" init --quiet
        git -C "$source" remote add origin "$repository"
    fi

    if [[ -n "$sparse_path" ]]; then
        git -C "$source" sparse-checkout init --cone
        git -C "$source" sparse-checkout set "$sparse_path"
        git -C "$source" fetch --depth 1 --filter=blob:none origin "$commit"
    else
        git -C "$source" fetch --depth 1 origin "$commit"
    fi
    git -C "$source" checkout --quiet --detach FETCH_HEAD
    check_git_source "$section" "$label"
}

check_tflite_reference() {
    check_git_source tensorflow TensorFlow
    check_git_source gemmlowp gemmlowp
    check_git_source ruy Ruy
}

fetch_tflite_reference() {
    local manifest="$PROJECT_ROOT/deps/manifests/tflite-reference.toml"
    local sparse_path
    sparse_path=$(toml_string "$manifest" tensorflow sparse_path)
    fetch_git_source tensorflow TensorFlow "$sparse_path"
    fetch_git_source gemmlowp gemmlowp
    fetch_git_source ruy Ruy
}

check_model() {
    local manifest="$PROJECT_ROOT/models/manifest.toml"
    local section path expected
    for section in efficientnet_lite0_int8 efficientnet_lite0_fp32; do
        path=$(resolve_project_path "$(toml_string "$manifest" "$section" path)")
        expected=$(toml_string "$manifest" "$section" sha256)
        verify_sha256 "$path" "$expected" || die "$section is missing or has the wrong SHA256; run 'make fetch-model'"
        printf 'OK model: %s\n' "${path#$PROJECT_ROOT/}"
    done

    path=$(resolve_project_path "$(toml_string "$manifest" efficientnet_lite0_pytorch path)")
    expected=$(toml_string "$manifest" efficientnet_lite0_pytorch weights_sha256)
    verify_sha256 "$path" "$expected" || die "PyTorch weights are missing or have the wrong SHA256; run 'make fetch-model'"
    printf 'OK model: %s\n' "${path#$PROJECT_ROOT/}"

    for section in phase1_1_test_image imagenet_labels; do
        path=$(resolve_project_path "$(toml_string "$manifest" "$section" path)")
        expected=$(toml_string "$manifest" "$section" sha256)
        verify_sha256 "$path" "$expected" || die "$section is missing or has the wrong SHA256"
        printf 'OK test data: %s\n' "${path#$PROJECT_ROOT/}"
    done
}

fetch_model() {
    local manifest="$PROJECT_ROOT/models/manifest.toml"
    local url archive_sha path model_sha archive staging candidate section
    local need_archive=false
    url=$(toml_string "$manifest" efficientnet_lite0_int8 source_archive_url)
    archive_sha=$(toml_string "$manifest" efficientnet_lite0_int8 source_archive_sha256)
    archive="$DOWNLOAD_DIR/efficientnet-lite0.tar.gz"

    for section in efficientnet_lite0_int8 efficientnet_lite0_fp32; do
        path=$(resolve_project_path "$(toml_string "$manifest" "$section" path)")
        model_sha=$(toml_string "$manifest" "$section" sha256)
        verify_sha256 "$path" "$model_sha" || need_archive=true
    done

    if $need_archive; then
        download "$url" "$archive_sha" "$archive"
        mkdir -p "$PROJECT_ROOT/deps/.work"
        staging=$(mktemp -d "$PROJECT_ROOT/deps/.work/model.XXXXXX")
        trap 'rm -rf -- "$staging"' RETURN
        tar -xzf "$archive" -C "$staging"
        for section in efficientnet_lite0_int8 efficientnet_lite0_fp32; do
            path=$(resolve_project_path "$(toml_string "$manifest" "$section" path)")
            model_sha=$(toml_string "$manifest" "$section" sha256)
            candidate=$(find "$staging" -type f -name "$(basename -- "$path")" -print -quit)
            [[ -n "$candidate" ]] || die "$(basename -- "$path") was not found in the verified archive"
            verify_sha256 "$candidate" "$model_sha" || die "extracted $(basename -- "$path") has the wrong SHA256"
            mkdir -p "$(dirname -- "$path")"
            mv -f -- "$candidate" "$path"
        done
        rm -rf -- "$staging"
        trap - RETURN
    fi

    path=$(resolve_project_path "$(toml_string "$manifest" efficientnet_lite0_pytorch path)")
    model_sha=$(toml_string "$manifest" efficientnet_lite0_pytorch weights_sha256)
    url=$(toml_string "$manifest" efficientnet_lite0_pytorch weights_url)
    if ! verify_sha256 "$path" "$model_sha"; then
        download "$url" "$model_sha" "$DOWNLOAD_DIR/$(basename -- "$path")"
        mkdir -p "$(dirname -- "$path")"
        cp -- "$DOWNLOAD_DIR/$(basename -- "$path")" "$path"
    fi
    check_model
}

if [[ ${1:-} == "--check" ]]; then
    CHECK_ONLY=true
    shift
fi

target=${1:-}
[[ $# -eq 1 ]] || { usage >&2; exit 2; }

case "$target" in
    toolchain)
        if $CHECK_ONLY; then check_toolchain; else fetch_toolchain; fi
        ;;
    spike)
        if $CHECK_ONLY; then check_spike; else fetch_spike; fi
        ;;
    model)
        if $CHECK_ONLY; then check_model; else fetch_model; fi
        ;;
    tflite-reference)
        if $CHECK_ONLY; then check_tflite_reference; else fetch_tflite_reference; fi
        ;;
    all)
        if $CHECK_ONLY; then
            check_toolchain
            check_spike
            check_model
            check_tflite_reference
        else
            fetch_toolchain
            fetch_spike
            fetch_model
            fetch_tflite_reference
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac
