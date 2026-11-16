#!/usr/bin/env python3
"""Hardware-free regression for the configurable torso-position stop."""
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = r'''
#include "beyondmimic/position_error_guard.h"
#include <cassert>
#include <limits>
#include <string>
#include <iostream>
template<class F> void rejects(F fn) {
    bool rejected=false;
    try { fn(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    using beyondmimic::check_anchor_position_error;
    assert(!check_anchor_position_error(0.1f,0.35f));
    assert(!check_anchor_position_error(0.35f,0.35f));
    rejects([&] { check_anchor_position_error(0.36f,0.35f); });
    rejects([&] { check_anchor_position_error(2,0.35f,true); });
    assert(check_anchor_position_error(0.36f,0.35f,false));
    assert(check_anchor_position_error(2,0.35f,false));
    assert(!check_anchor_position_error(0.1f,0.35f,false));
    for(bool enforce : {true,false}) {
        rejects([&] { check_anchor_position_error(std::numeric_limits<float>::quiet_NaN(),0.35f,enforce); });
        rejects([&] { check_anchor_position_error(std::numeric_limits<float>::infinity(),0.35f,enforce); });
        rejects([&] { check_anchor_position_error(-1,0.35f,enforce); });
        rejects([&] { check_anchor_position_error(0.1f,0,enforce); });
    }
    std::cout << "PASS: default enforcement, threshold boundary, log-only mode and invalid-data rejection" << std::endl;
}
'''


def main():
    compiler = shutil.which("c++")
    if not compiler:
        raise SystemExit("C++ compiler is required")
    with tempfile.TemporaryDirectory(prefix="g1-position-guard-test-") as directory:
        source = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        source.write_text(SOURCE)
        include = Path(__file__).resolve().parents[1] / "deploy/include"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-I", str(include),
                        str(source), "-o", str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=5)


if __name__ == "__main__":
    main()
