// Exact production Core + ONNX Runner, without SDK/DDS or a connected robot.
#include "velocity11k/runner.h"
#include <iomanip>
#include <iostream>

template<class Values> void read(Values& values) {
    for(auto& value:values) if(!(std::cin>>value)) throw std::runtime_error("Incomplete offline snapshot");
}
template<class Values> void write(const Values& values) {
    bool first=true;
    for(auto value:values) { if(!first) std::cout << ' '; first=false; std::cout << value; }
    std::cout << std::endl;
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--version") {
            std::cout<<OrtGetApiBase()->GetVersionString()<<'\n';
            return 0;
        }
        if(argc!=2) throw std::runtime_error("Usage: velocity11k_offline PATH_TO_11K_POLICY_ONNX");
        velocity11k::Runner runner(std::filesystem::u8path(argv[1]));
        velocity11k::Core core;
        std::cout << std::setprecision(9);
        int reset;
        while(std::cin>>reset) {
            velocity11k::Snapshot snapshot;
            read(snapshot.quat); read(snapshot.omega); read(snapshot.q); read(snapshot.dq); read(snapshot.command);
            if(reset) core.reset(snapshot);
            const auto obs=core.observe(snapshot);
            const auto actions=runner.act(obs);
            write(obs); write(actions); write(core.targets(actions));
        }
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
