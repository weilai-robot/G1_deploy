#!/usr/bin/env bash
# Build the HugWBC variant (r41 default). Wraps build_g1_deploy.sh with the
# explicit HugWBC config; the HugWBC package checker runs, BeyondMimic's does not.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:-r41}"
case "$variant" in
  r41|r42) ;;
  *) echo "Usage: build_g1_hugwbc.sh [r41|r42]" >&2; exit 1 ;;
esac
exec "$repo_dir/tools/build_g1_deploy.sh" --config \
  "$repo_dir/deploy/robots/g1_29dof/config/hugwbc_$variant.yaml"
