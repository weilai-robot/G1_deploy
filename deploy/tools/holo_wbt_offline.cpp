// 与真机状态共用同一份观测拼装/推理/限幅代码，无 DDS。
// JSON-lines 输入输出，便于在 MuJoCo 或 Python 侧做逐帧等价性比对。
#include "holo_wbt/package.h"
#include <iomanip>
#include <iostream>

template <class Values>
void array_json(const Values& values) {
    bool first = true;
    std::cout << '[';
    for (auto v : values) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << v;
    }
    std::cout << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: holo_wbt_offline PACKAGE_DIRECTORY");
        const std::filesystem::path package = argv[1];
        auto cfg = holowbt::load_config(package / "deploy.yaml");
        holowbt::Core core(cfg);
        holowbt::Motion motion((package / "reference.bin").string());
        holowbt::Runner runner(package / "policy.onnx");
        bool initialised = false;
        std::string line;
        std::cout << std::setprecision(9);
        while (std::getline(std::cin, line)) {
            auto input = YAML::Load(line);
            holowbt::Snapshot s;
            s.q = holowbt::read_array<29>(input["q"]);
            s.dq = holowbt::read_array<29>(input["dq"]);
            auto q = holowbt::read_array<4>(input["quat"]);
            auto w = holowbt::read_array<3>(input["omega"]);
            s.orientation = holowbt::Quat(q[0], q[1], q[2], q[3]);
            s.angular_velocity = holowbt::Vec(w.data());
            auto ref = motion.frame(input["frame"].as<std::size_t>());
            if (!initialised || (input["reset"] && input["reset"].as<bool>())) {
                core.reset(s);
                initialised = true;
            }
            auto obs = core.observe(s, ref);
            auto actions = runner.act(obs);
            auto targets = core.targets(actions, s, input["max_speed"] ? input["max_speed"].as<float>() : 8.0f);
            std::cout << "{\"obs\":"; array_json(obs);
            std::cout << ",\"actions\":"; array_json(actions);
            std::cout << ",\"targets\":"; array_json(targets);
            std::cout << ",\"clipped\":" << core.clipped() << "}" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "holo_wbt offline error: " << e.what() << '\n';
        return 1;
    }
}
