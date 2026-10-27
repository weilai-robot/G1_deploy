// Runs the SAME observation, inference, limits and odometry as the real state,
// without DDS. JSON-lines stdin/stdout allows headless MuJoCo closed-loop tests.
#include "beyondmimic/package.h"
#include <iomanip>
#include <iostream>

template<class Values>
void array_json(const Values& values) {
    bool first=true;
    std::cout << '[';
    for(auto v:values) { if(!first) std::cout << ','; first=false; std::cout << v; }
    std::cout << ']';
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Usage: beyondmimic_offline PACKAGE_DIRECTORY");
        const std::filesystem::path package=argv[1];
        auto cfg=beyondmimic::load_config(package/"deploy.yaml");
        beyondmimic::Core core(cfg);
        beyondmimic::Motion motion((package/"reference.bin").string());
        beyondmimic::Runner runner(package/"policy.onnx");
        bool initialised=false;
        std::string line;
        std::cout << std::setprecision(9);
        while(std::getline(std::cin,line)) {
            auto input=YAML::Load(line);
            beyondmimic::Snapshot s;
            s.q=beyondmimic::read_array<29>(input["q"]);
            s.dq=beyondmimic::read_array<29>(input["dq"]);
            auto q=beyondmimic::read_array<4>(input["quat"]);
            auto w=beyondmimic::read_array<3>(input["omega"]);
            s.orientation=beyondmimic::Quat(q[0],q[1],q[2],q[3]);
            s.angular_velocity=beyondmimic::Vec(w.data());
            auto ref=motion.frame(input["frame"].as<std::size_t>());
            if(!initialised || (input["reset"] && input["reset"].as<bool>())) {
                core.reset(s,ref); initialised=true;
            }
            beyondmimic::Vec position,velocity;
            const bool truth=bool(input["position"]);
            if(truth) {
                auto p=beyondmimic::read_array<3>(input["position"]);
                auto v=beyondmimic::read_array<3>(input["velocity_b"]);
                position=beyondmimic::Vec(p.data()); velocity=beyondmimic::Vec(v.data());
            }
            auto obs=core.observe(s,ref,true,truth?&position:nullptr,truth?&velocity:nullptr);
            auto actions=runner.act(obs);
            auto targets=core.targets(actions,s,input["max_speed"]?input["max_speed"].as<float>():8.0f);
            std::cout << "{\"obs\":"; array_json(obs);
            std::cout << ",\"actions\":"; array_json(actions);
            std::cout << ",\"targets\":"; array_json(targets);
            std::cout << ",\"position\":"; array_json(core.position());
            std::cout << ",\"velocity_b\":"; array_json(core.velocity_b());
            std::cout << ",\"clipped\":" << core.clipped() << "}" << std::endl;
        }
    } catch(const std::exception& e) {
        std::cerr << "BeyondMimic offline error: " << e.what() << '\n';
        return 1;
    }
}
