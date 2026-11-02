// Hardware-free HugWBC replay: the SAME mapping, clock, observation, hybrid
// action and command code as State_HugWBC, without DDS. JSON-lines stdin/
// stdout allows headless parity tests against the Python reference and later
// closed-loop simulation of the deployment core. No robot commands are sent.
#include "hugwbc/package.h"
#include "hugwbc/runner.h"
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
        if (argc != 2) throw std::runtime_error("Usage: hugwbc_offline PACKAGE_DIRECTORY");
        const std::filesystem::path package = argv[1];
        auto loaded = hugwbc::load_package(package);
        hugwbc::Core core(loaded.cfg);
        hugwbc::Runner runner(package / "policy.onnx", loaded.policy_sha256);
        bool initialised = false;
        std::string line;
        std::cout << std::setprecision(9);
        while (std::getline(std::cin, line)) {
            auto input = YAML::Load(line);
            hugwbc::Snapshot s;
            s.q = hugwbc::read_array<hugwbc::SDK>(input["q"]);
            s.dq = hugwbc::read_array<hugwbc::SDK>(input["dq"]);
            auto quat = hugwbc::read_array<4>(input["quat"]);
            auto omega = hugwbc::read_array<3>(input["omega"]);
            s.orientation = hugwbc::Quat(quat[0], quat[1], quat[2], quat[3]);  // wxyz
            s.angular_velocity = hugwbc::Vec(omega[0], omega[1], omega[2]);
            auto command = hugwbc::read_array<hugwbc::CMD>(input["cmd"]);
            const bool standing = input["standing"] ? input["standing"].as<bool>() : false;
            const bool reset = !initialised || (input["reset"] && input["reset"].as<bool>());
            if (reset) core.reset();
            initialised = true;
            core.prepare(s, command, standing);
            const auto raw = runner.act(core.history());
            auto tick = core.complete(s, command, standing, raw);
            std::cout << "{\"index\":" << tick.index;
            std::cout << ",\"obs\":"; array_json(tick.frame.obs);
            std::cout << ",\"history\":"; array_json(tick.history);
            std::cout << ",\"g\":" << tick.frame.g;
            std::cout << ",\"phase_left\":" << tick.frame.phase_left;
            std::cout << ",\"phase_right\":" << tick.frame.phase_right;
            std::cout << ",\"clock_left\":" << tick.frame.clock_left;
            std::cout << ",\"clock_right\":" << tick.frame.clock_right;
            std::cout << ",\"raw\":"; array_json(tick.raw_action);
            std::cout << ",\"hybrid\":"; array_json(tick.hybrid_action);
            std::cout << ",\"clipped\":"; array_json(tick.clipped_action);
            std::cout << ",\"q_target\":"; array_json(tick.q_target);
            std::cout << ",\"arm_reference\":"; array_json(tick.arm_reference);
            std::cout << ",\"sdk_targets\":"; array_json(tick.sdk_targets);
            std::cout << ",\"limited\":" << tick.limited_count << "}" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "HugWBC offline error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
