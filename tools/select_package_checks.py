#!/usr/bin/env python3
"""Print the package-checker commands required by a deploy FSM config.

Stdout lines are '<script> [args...]':
  check_beyondmimic_package.py [package]  when BeyondMimic is enabled
  check_hugwbc_package.py [package]       per HugWBC policy_dir
Configs enabling neither (e.g. the default Velocity config) print nothing.
policy_dir is resolved relative to the robot project directory (the parent of
the config directory), matching param::proj_dir in the C++ loader.
"""
import sys
from pathlib import Path

import yaml


def main():
    config = Path(sys.argv[1])
    if not config.is_file():
        raise SystemExit(f"Config not found: {config}")
    cfg = yaml.safe_load(config.read_text()) or {}
    fsm = cfg.get("FSM") or {}
    fsms = fsm.get("_") or {}
    project_dir = config.parent.parent
    for name, entry in fsms.items():
        if not isinstance(entry, dict):
            continue
        state_type = entry.get("type", name)
        state_cfg = fsm.get(name) or {}
        if state_type == "BeyondMimic":
            package = project_dir / state_cfg.get(
                "policy_dir", "config/policy/beyondmimic/dance_zui")
            print(f"check_beyondmimic_package.py {package.resolve()}")
        elif state_type == "HugWBC":
            if "policy_dir" not in state_cfg:
                raise SystemExit(f"FSM '{name}' has no policy_dir")
            package = (project_dir / state_cfg["policy_dir"]).resolve()
            print(f"check_hugwbc_package.py {package}")


if __name__ == "__main__":
    main()
