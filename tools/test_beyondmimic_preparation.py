#!/usr/bin/env python3
"""Compile/run the actual preparation planner without DDS, SDK or a robot."""
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = r'''
#include "beyondmimic/preparation.h"
#include <cassert>
#include <iostream>
#include <limits>
using Config = beyondmimic::PreparationConfig<2>;
using Planner = beyondmimic::Preparation<2>;
using Values = Planner::Values;
Config config() {
    Config c;
    c.standing_kp={150,100}; c.standing_kd={3,2};
    c.policy_kp={28.5f,40}; c.policy_kd={1.81f,2.5f};
    c.lower={-2,-2}; c.upper={2,2}; c.effort={100,100};
    return c;
}
template<class F> void rejects(F fn) {
    bool rejected=false;
    try { fn(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    auto c=config();
    Planner p(c,{0,0},{0.8f,-0.6f});
    Values actual{}, velocity{};
    bool ready=false;
    for(int n=0;n<220 && !ready;++n) {
        const float t=n*c.dt;
        auto cmd=p.update(t,actual,velocity);
        for(int i=0;i<2;++i) {
            assert(std::abs(cmd.q[i]-actual[i])<=c.max_target_speed*c.dt+1e-6f);
            assert(std::abs(cmd.kp[i]*(cmd.q[i]-actual[i])-cmd.kd[i]*velocity[i])<=c.effort[i]+1e-4f);
            if(t<=3) assert(cmd.kp[i]==c.standing_kp[i]);
            if(t>=3.5f) assert(std::abs(cmd.kp[i]-c.policy_kp[i])<1e-4f);
        }
        if(t<3.69f) assert(!cmd.ready);
        actual=cmd.q; ready=cmd.ready;
    }
    assert(ready); // posture + gain blend + measured settling completed

    // Fast handoff retains the measured pose/velocity gate but does not hold
    // the motion's first frame for the former 0.2-second confirmation period.
    c=config(); c.pose_duration=0.6f; c.gain_duration=0.1f;
    c.max_target_speed=2; c.settle_duration=0.04f; c.settle_timeout=0.5f;
    Planner fast(c,{0,0},{0.9f,-0.6f}); actual={0,0}; velocity={0,0}; ready=false;
    assert(fast.pose_duration()>=0.674f); // cannot outrun the target speed limit
    for(int n=0;n<55 && !ready;++n) {
        const float t=n*c.dt;
        auto cmd=fast.update(t,actual,velocity);
        for(int i=0;i<2;++i) {
            assert(std::abs(cmd.q[i]-actual[i])<=c.max_target_speed*c.dt+1e-6f);
            assert(std::abs(cmd.kp[i]*(cmd.q[i]-actual[i])-cmd.kd[i]*velocity[i])<=c.effort[i]+1e-4f);
            velocity[i]=(cmd.q[i]-actual[i])/c.dt;
        }
        if(t<fast.pose_duration()+c.gain_duration+c.settle_duration-1e-5f) assert(!cmd.ready);
        actual=cmd.q; ready=cmd.ready;
    }
    assert(ready);
    Planner fast_stuck(c,{0,0},{0.9f,0});
    rejects([&] { for(int n=0;n<100;++n) assert(!fast_stuck.update(n*c.dt,{0,0},{0,0}).ready); });
    Planner fast_settling(c,{0,0},{0,0});
    assert(!fast_settling.update(0.71f,{0,0},{0,0}).ready);
    assert(!fast_settling.update(0.73f,{0,0},{1,0}).ready);
    assert(!fast_settling.update(0.75f,{0,0},{0,0}).ready);
    assert(!fast_settling.update(0.77f,{0,0},{0,0}).ready);
    assert(fast_settling.update(0.81f,{0,0},{0,0}).ready);

    // Diagnostic mode uses high destination PD and keeps updating the same
    // planner after readiness; it must not blend down to the trained gains.
    c.policy_kp={150,150}; c.policy_kd=c.standing_kd;
    Planner hold(c,{0,0},{0.9f,-0.6f}); actual={0,0}; velocity={0,0}; ready=false;
    for(int n=0;n<500;++n) {
        auto cmd=hold.update(n*c.dt,actual,velocity);
        for(int i=0;i<2;++i) {
            if(n*c.dt>=hold.pose_duration()+c.gain_duration)
                assert(cmd.kp[i]==150 && cmd.kd[i]==c.standing_kd[i]);
            velocity[i]=(cmd.q[i]-actual[i])/c.dt;
        }
        if(ready) assert(cmd.ready);
        ready=cmd.ready; actual=cmd.q;
    }
    assert(ready);
    rejects([&] { hold.update(10,{actual[0]+0.5f,actual[1]},{0,0}); });
    c.settle_duration=0;
    rejects([&] { Planner bad(c,{0,0},{0.2f,0}); });
    c.settle_duration=0.01f;
    rejects([&] { Planner bad(c,{0,0},{0.2f,0}); });
    c.settle_duration=c.settle_timeout;
    rejects([&] { Planner bad(c,{0,0},{0.2f,0}); });

    // A reference close to a hard joint limit is reachable within the pose
    // tolerance, while commanded targets retain the deployment's 0.01 margin.
    c=config(); Planner margin(c,{0,0},{1.999f,0}); actual={0,0}; ready=false;
    for(int n=0;n<500 && !ready;++n) {
        auto cmd=margin.update(n*c.dt,actual,{0,0});
        assert(cmd.q[0]<=1.990001f); actual=cmd.q; ready=cmd.ready;
    }
    assert(ready);

    c=config(); c.pose_duration=1; c.max_target_speed=0.2f;
    Planner slow(c,{0,0},{1,0});
    assert(std::abs(slow.pose_duration()-7.5f)<1e-5f);

    c=config(); c.max_error=0.01f;
    Planner stuck(c,{0,0},{0.8f,0});
    rejects([&] { for(int n=0;n<300;++n) assert(!stuck.update(n*c.dt,{0,0},{0,0}).ready); });

    c=config(); Planner moving(c,{0,0},{0.2f,0}); actual={0,0};
    rejects([&] { for(int n=0;n<300;++n) {
        auto cmd=moving.update(n*c.dt,actual,{1,0}); assert(!cmd.ready); actual=cmd.q;
    } });

    c=config(); c.effort={0.01f,0.01f}; Planner conflict(c,{0,0},{0.2f,0});
    rejects([&] { conflict.update(0,{0.5f,0},{0,0}); });
    rejects([&] { Planner bad(config(),{0,0},{3,0}); });
    c=config(); c.standing_kp[0]=-1;
    rejects([&] { Planner bad(c,{0,0},{0.2f,0}); });
    c=config(); c.pose_duration=std::numeric_limits<float>::infinity();
    rejects([&] { Planner bad(c,{0,0},{0.2f,0}); });
    Planner invalid(config(),{0,0},{0.2f,0});
    rejects([&] { invalid.update(0,{0,0},{std::numeric_limits<float>::quiet_NaN(),0}); });

    Planner settling(config(),{0,0},{0,0});
    assert(!settling.update(3.5f,{0,0},{0,0}).ready);
    assert(!settling.update(3.6f,{0,0},{1,0}).ready); // unstable feedback resets dwell
    assert(!settling.update(3.7f,{0,0},{0,0}).ready);
    assert(!settling.update(3.8f,{0,0},{0,0}).ready);
    assert(settling.update(3.92f,{0,0},{0,0}).ready);
    std::cout << "PASS: pose/gain sequencing, measured settling, speed/effort bounds, "
                 "timeouts and invalid inputs" << std::endl;
}
'''


def main():
    compiler = shutil.which("c++")
    if not compiler:
        raise SystemExit("C++ compiler is required")
    with tempfile.TemporaryDirectory(prefix="g1-preparation-test-") as directory:
        source = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        source.write_text(SOURCE)
        include = Path(__file__).resolve().parents[1] / "deploy/include"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-I", str(include),
                        str(source), "-o", str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=5)


if __name__ == "__main__":
    main()
