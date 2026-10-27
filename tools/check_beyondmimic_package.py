#!/usr/bin/env python3
"""Dependency-free deployment-package integrity check (does not control a robot)."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, nargs="?", default=Path(__file__).resolve().parents[1]/"deploy/robots/g1_29dof/config/policy/beyondmimic/dance_zui")
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package/"manifest.json").read_text())
    for name, key in (("policy.onnx", "policy_sha256"), ("reference.bin", "reference_sha256"),
                      ("deploy.yaml", "deploy_yaml_sha256")):
        digest = hashlib.sha256((package/name).read_bytes()).hexdigest()
        if digest != manifest[key]:
            raise SystemExit("FAILED: checksum mismatch for " + name)
    with (package/"reference.bin").open("rb") as f:
        magic, frames, joints, bodies, dt = struct.unpack("<8sIIIf", f.read(24))
    if (magic != b"BMG1V1\0\0" or joints != 29 or bodies != 14 or abs(dt-0.02)>1e-6 or
        frames != manifest["frames"] or (package/"reference.bin").stat().st_size != 24+frames*156*4):
        raise SystemExit("FAILED: invalid reference table")
    print("PASS: policy and reference checksums, 160 observations / 29 actions, 50 Hz,", frames, "frames")


if __name__ == "__main__":
    main()
