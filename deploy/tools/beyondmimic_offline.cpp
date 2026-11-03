// Runs the SAME observation, inference, limits and odometry as the real state,
// without DDS. JSON-lines stdin/stdout allows headless MuJoCo closed-loop tests.
#include "beyondmimic/package.h"
#include "beyondmimic/cyclic.h"
#include <iomanip>
#include <iostream>

template<class Values>
void array_json(const Values& values) {
    bool first=true;
    std::cout << '[';
    for(std::size_t i=0;i<static_cast<std::size_t>(values.size());++i) {
        if(!first) std::cout << ','; first=false; std::cout << values[i];
    }
    std::cout << ']';
}
int main(int argc,char** argv) {
    try {
        if(argc==3 && std::string(argv[1])=="--check-walk-config") {
            const auto config_path=std::filesystem::absolute(argv[2]);
            const auto fsm=YAML::LoadFile(config_path.string());
            const auto walking=fsm["FSM"]["BeyondMimicWalk"];
            const auto path=config_path.parent_path().parent_path()/walking["policy_dir"].as<std::string>();
            const auto package=YAML::LoadFile((path/"deploy.yaml").string());
            const auto cfg=beyondmimic::load_config(path/"deploy.yaml");
            const auto rate=walking["max_target_speed"].as<float>();
            beyondmimic::validate_trained_target_rate(package,rate);
            beyondmimic::CyclicGait gait((path/"gait.bin").string());
            gait.reset(walking["initial_support_index"].as<std::size_t>(0));
            if(std::abs(gait.dt()-cfg.dt)>1e-6f)throw std::runtime_error("Walking gait/control clock mismatch");
            const auto ref=gait.reference();
            const auto stand=fsm["FSM"]["FixStand"];
            const auto q=beyondmimic::read_array<29>(stand["qs"][1]);
            const auto kp=beyondmimic::read_array<29>(stand["kp"]);
            const auto kd=beyondmimic::read_array<29>(stand["kd"]);
            for(int i=0;i<beyondmimic::J;++i) {
                const auto sdk=cfg.sdk_indices[i];
                if(q[sdk]!=ref.q[i] || kp[sdk]!=cfg.kp[i] || kd[sdk]!=cfg.kd[i])
                    throw std::runtime_error("FixStand differs from selected model support/PD");
            }
            const auto dimension=package["observation_dim"].as<int>();
            beyondmimic::Runner runner(path/"policy.onnx",dimension);
            const auto manifest=YAML::LoadFile((path/"manifest.json").string());
            std::cout << "{\"model_id\":\"" << package["model_id"].as<std::string>()
                      << "\",\"checkpoint_sha256\":\"" << manifest["checkpoint_sha256"].as<std::string>()
                      << "\",\"observation_dim\":" << dimension
                      << ",\"fixstand_matches_initial_support\":true,\"onnx_session_created\":true,"
                      << "\"trained_target_contract_checked\":" << (package["deployment_target_contract"]?"true":"false")
                      << ",\"robot_commands_sent\":false}" << std::endl;
            return 0;
        }
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
