// Hardware-free test of the exact history/mapping code used by State_Velocity11k.
// Each request is: reset quat[4] gyro[3] q_sdk[29] dq_sdk[29] command[3].
// Reply: obs[480]. Then send raw action[29]; reply: target_q_sdk[29].
#include "velocity11k/core.h"
#include <iomanip>
#include <iostream>

template<class Values> void read(Values& values) {
    for(auto& value:values) if(!(std::cin>>value)) throw std::runtime_error("Incomplete snapshot/action input");
}
template<class Values> void write(const Values& values) {
    bool first=true;
    for(auto value:values) { if(!first) std::cout << ' '; first=false; std::cout << value; }
    std::cout << std::endl;
}
int main() {
    try {
        std::cout << std::setprecision(9);
        velocity11k::Core core;
        int reset;
        while(std::cin>>reset) {
            velocity11k::Snapshot snapshot;
            read(snapshot.quat); read(snapshot.omega); read(snapshot.q); read(snapshot.dq); read(snapshot.command);
            if(reset) core.reset(snapshot);
            write(core.observe(snapshot));
            velocity11k::Joints action;
            read(action);
            write(core.targets(action));
        }
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
