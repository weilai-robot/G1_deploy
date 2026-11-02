#!/usr/bin/env bash
# Build on the Linux deployment computer. Does not start robot control.
# Usage: build_g1_deploy.sh [--config deploy/robots/g1_29dof/config/<name>.yaml]
# The package checker is selected by the FSMs actually enabled in the config,
# so building a HugWBC config no longer silently runs the BeyondMimic check.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$repo_dir/deploy/robots/g1_29dof/config/config.yaml"
if [[ "${1:-}" == "--config" ]]; then
  [[ -n "${2:-}" ]] || { echo "--config requires a path" >&2; exit 1; }
  config="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
elif [[ -n "${1:-}" ]]; then
  config="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
fi
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }
if [[ "$(uname -s)" != Linux ]]; then
  echo "Build the robot controller on Linux (the G1 onboard computer or a Linux PC)." >&2
  exit 1
fi
case "$(uname -m)" in
  aarch64|arm64) runtime_arch=aarch64 ;;
  x86_64|amd64) runtime_arch=x64 ;;
  *) echo "Unsupported CPU architecture" >&2; exit 1 ;;
esac
runtime_version=1.22.0
runtime_dir="${ONNXRUNTIME_ROOT:-$repo_dir/deploy/thirdparty/onnxruntime-linux-$runtime_arch-$runtime_version}"
if [[ ! -f "$runtime_dir/include/onnxruntime_cxx_api.h" || ! -f "$runtime_dir/lib/libonnxruntime.so.$runtime_version" ]]; then
  mkdir -p "$(dirname "$runtime_dir")"
  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  curl --fail --location --retry 2 "https://github.com/microsoft/onnxruntime/releases/download/v$runtime_version/onnxruntime-linux-$runtime_arch-$runtime_version.tgz" -o "$download_dir/runtime.tgz"
  tar -xzf "$download_dir/runtime.tgz" -C "$download_dir"
  if [[ -e "$runtime_dir" ]]; then
    echo "Incomplete runtime directory exists: $runtime_dir; repair it or choose ONNXRUNTIME_ROOT." >&2
    exit 1
  fi
  mv "$download_dir/onnxruntime-linux-$runtime_arch-$runtime_version" "$runtime_dir"
fi
# Select package checks by the enabled FSM states of this config. The
# selector prints TAB-separated '<checker> <package>'; do not re-quote the
# whole line as one filename. CR is stripped in case the checkout carries
# CRLF line endings (e.g. Windows working copies pushed to Linux builders).
python3 "$repo_dir/tools/select_package_checks.py" "$config" | tr -d '\r' \
| while IFS=$'\t' read -r checker package; do
  [[ -n "$checker" ]] || continue
  if [[ -n "${package:-}" ]]; then
    python3 "$repo_dir/tools/$checker" "$package"
  else
    python3 "$repo_dir/tools/$checker"
  fi
done
cmake -S "$repo_dir/deploy/robots/g1_29dof" -B "$repo_dir/deploy/robots/g1_29dof/build" \
  -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$runtime_dir" \
  -DONNXRUNTIME_INCLUDE_DIR="$runtime_dir/include" \
  -DONNXRUNTIME_LIBRARY="$runtime_dir/lib/libonnxruntime.so.$runtime_version" \
  -DCMAKE_PREFIX_PATH="${UNITREE_SDK_PREFIX:-/usr/local}"
cmake --build "$repo_dir/deploy/robots/g1_29dof/build" -j "${BUILD_JOBS:-2}"
echo "Built controller; no robot commands were sent. Config: $config"
