#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Install NVIDIA cuDSS into a user/project-space prefix.

Usage:
  tools/install_cudss.sh [--prefix DIR] [--package NAME] [--tarball FILE]

Default prefix:
  <repo-root>/extern/cudss

Install modes:
  1. Pip wheel mode, default:
       Tries the wheel matching the detected CUDA major version first, then
       falls back to the other CUDA wheel and nvidia-cudss unless --package NAME
       is provided.

  2. Tarball mode:
       Use --tarball /path/to/cudss-linux-*.tar.* after downloading the package
       from NVIDIA manually.

After installation, build with:
  make -C examples USE_CUDA=1

The makefile auto-detects the default checkout-local install path:
  extern/cudss/python/nvidia/cu12
  extern/cudss/python/nvidia/cu13

If using a pip wheel with only libcudss.so.0 and no libcudss.so symlink, the
project makefile will link it with -l:libcudss.so.0 automatically.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

prefix="${repo_root}/extern/cudss"
package=""
tarball=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      prefix="$2"
      shift 2
      ;;
    --package)
      package="$2"
      shift 2
      ;;
    --tarball)
      tarball="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd)"

find_cudss_root() {
  local root="$1"
  local header
  header="$(find "$root" -path '*/include/cudss.h' -print -quit 2>/dev/null || true)"
  if [[ -z "$header" ]]; then
    return 1
  fi
  dirname "$(dirname "$header")"
}

write_env_file() {
  local cudss_root="$1"
  local libdir=""
  if compgen -G "$cudss_root/lib64/libcudss.so*" >/dev/null; then
    libdir="$cudss_root/lib64"
  elif compgen -G "$cudss_root/lib/libcudss.so*" >/dev/null; then
    libdir="$cudss_root/lib"
  else
    libdir="$(find "$cudss_root" -type f \( -name 'libcudss.so' -o -name 'libcudss.so.0' \) -print -quit 2>/dev/null || true)"
    libdir="$(dirname "$libdir")"
  fi

  if [[ -z "$libdir" || "$libdir" == "." || ! -d "$libdir" ]]; then
    echo "[ERROR] Could not find libcudss under $cudss_root" >&2
    exit 1
  fi

  cat > "$prefix/cudss-env.sh" <<EOF
export CUDSS_PATH="$cudss_root"
export CUDSS_LIBDIR="$libdir"
export LD_LIBRARY_PATH="$libdir:\${LD_LIBRARY_PATH:-}"
EOF

  echo ""
  echo "[OK] cuDSS root: $cudss_root"
  echo "[OK] cuDSS env : $prefix/cudss-env.sh"
  echo ""
  echo "Use:"
  echo "  make -C examples USE_CUDA=1"
  echo ""
  echo "If this cuDSS prefix is not the default sibling path for your checkout, use:"
  echo "  source $prefix/cudss-env.sh"
  echo "  make -C examples USE_CUDA=1 CUDSS_PATH=\$CUDSS_PATH CUDSS_LIBDIR=\$CUDSS_LIBDIR"
}

if [[ -n "$tarball" ]]; then
  if [[ ! -f "$tarball" ]]; then
    echo "[ERROR] Tarball not found: $tarball" >&2
    exit 1
  fi
  rm -rf "$prefix/extract"
  mkdir -p "$prefix/extract"
  tar -xf "$tarball" -C "$prefix/extract"
  cudss_root="$(find_cudss_root "$prefix/extract")" || {
    echo "[ERROR] Could not find include/cudss.h after extracting $tarball" >&2
    exit 1
  }
  write_env_file "$cudss_root"
  exit 0
fi

python_bin="${PYTHON:-python3}"
target="$prefix/python"
mkdir -p "$target"

detect_cuda_major() {
  local nvcc_bin=""
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/cuda/bin/nvcc" ]]; then
    nvcc_bin="${CUDA_PATH}/cuda/bin/nvcc"
  elif [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    nvcc_bin="${CUDA_PATH}/bin/nvcc"
  fi

  if [[ -z "$nvcc_bin" ]]; then
    return 0
  fi

  "$nvcc_bin" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1
}

packages=()
if [[ -n "$package" ]]; then
  packages=("$package")
else
  cuda_major="$(detect_cuda_major || true)"
  if [[ "$cuda_major" == "13" ]]; then
    packages=("nvidia-cudss-cu13" "nvidia-cudss-cu12" "nvidia-cudss")
  else
    packages=("nvidia-cudss-cu12" "nvidia-cudss-cu13" "nvidia-cudss")
  fi
fi

install_ok=0
for pkg in "${packages[@]}"; do
  echo "[INFO] Trying pip package: $pkg"
  if "$python_bin" -m pip install --upgrade --target "$target" "$pkg"; then
    install_ok=1
    break
  fi
done

if [[ "$install_ok" -ne 1 ]]; then
  cat >&2 <<'EOF'
[ERROR] Could not install cuDSS from pip.

Fallback:
  1. Download the Linux cuDSS archive from NVIDIA:
       https://developer.nvidia.com/cudss-downloads
  2. Copy it to Perlmutter.
  3. Run:
       tools/install_cudss.sh --prefix /global/cfs/cdirs/.../cudss --tarball /path/to/archive.tar.*
EOF
  exit 1
fi

cudss_root="$(find_cudss_root "$target")" || {
  echo "[ERROR] pip install completed, but include/cudss.h was not found under $target" >&2
  exit 1
}

write_env_file "$cudss_root"

