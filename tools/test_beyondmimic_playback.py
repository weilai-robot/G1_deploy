#!/usr/bin/env python3
"""Hardware-free checks for full playback and starting from FixStand."""
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = r'''
#include "beyondmimic/playback.h"
#include <cassert>
#include <iostream>
#include <limits>
template<class F> void rejects(F fn) {
    bool rejected=false;
    try { fn(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    using beyondmimic::playback_range;
    // Default full mode must include the last frame, independent of a legacy
    // duration setting and its floating-point duration-to-frame conversion.
    const auto zui=playback_range(11384,0.02f);
    assert(zui.first==0 && zui.end==11384);
    assert(std::abs((zui.end-zui.first)*0.02-227.68)<1e-5);
    const auto guohua=playback_range(2400,0.02f,0,true,30);
    assert(guohua.first==0 && guohua.end==2400);
    assert(playback_range(2400,0.02f,100).end==2400);
    assert(playback_range(11384,0.02f,0,false,30).end==1500);
    assert(playback_range(2400,0.02f,2399,false,30).end==2400);
    rejects([&] { playback_range(2400,0.02f,2400); });
    rejects([&] { playback_range(2400,0.02f,0,false,0); });
    rejects([&] { playback_range(2400,0.02f,0,false,0.001f); });
    rejects([&] { playback_range(2400,0.02f,0,false,301); });
    rejects([&] { playback_range(2400,std::numeric_limits<float>::quiet_NaN()); });

    const std::array<float,3> standing{0.2f,-0.3f,0.6f};
    const std::array<float,3> reference{1.5f,0.2f,-0.3f};
    const std::array<int,3> ids{2,0,1};
    // The reference differs by 0.9 rad from standing in SDK joint 2. Direct
    // entry accepts the actual standing pose, without requiring frame 0 first.
    assert(beyondmimic::start_posture_error(standing,standing,reference,ids,true)==0);
    assert(beyondmimic::start_posture_error(standing,standing,reference,ids,false)>0.89f);
    auto bad_standing=standing; bad_standing[0]+=0.3f;
    assert(beyondmimic::start_posture_error(bad_standing,standing,reference,ids,true)>0.25f);
    const std::array<float,3> actual_reference{0.2f,-0.3f,1.5f};
    assert(beyondmimic::start_posture_error(actual_reference,standing,reference,ids,false)==0);
    auto invalid=standing; invalid[0]=std::numeric_limits<float>::quiet_NaN();
    rejects([&] { beyondmimic::start_posture_error(invalid,standing,reference,ids,true); });
    rejects([&] { beyondmimic::start_posture_error(standing,standing,reference,std::array<int,3>{2,0,3},true); });
    std::cout << "PASS: full frame ranges, explicit timed segments, SDK mapping, direct FixStand entry and invalid feedback" << std::endl;
}
'''


def main():
    compiler = shutil.which("c++")
    if not compiler:
        raise SystemExit("C++ compiler is required")
    with tempfile.TemporaryDirectory(prefix="g1-playback-test-") as directory:
        source = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        source.write_text(SOURCE)
        include = Path(__file__).resolve().parents[1] / "deploy/include"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-I", str(include),
                        str(source), "-o", str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=5)


if __name__ == "__main__":
    main()
