#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_BeyondMimic.h"
#include "State_BeyondMimicWalk.h"
#include <csignal>

volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) { stop_requested = 1; }

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        std::exit(1);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    const bool walk = bool(param::config["FSM"]["_"]["BeyondMimicWalk"]);
    const bool beyondmimic = bool(param::config["FSM"]["_"]["BeyondMimic"]);
    if((beyondmimic || walk) && vm["network"].as<std::string>().empty()) {
        std::cerr << "BeyondMimic requires an explicit --network (lo for simulation).\n";
        return 1;
    }
    // Parse --help before starting input threads. Walking uses only the remote;
    // retain keyboard input for the existing non-walking configurations.
    if(!walk) FSMState::keyboard = std::make_shared<Keyboard>();

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(vm["domain"].as<int>(), vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    if(walk) {
        std::cout << "Wait for model-specific FixStand; center sticks, then [R1 + X] -> BeyondMimicWalk.\n";
        std::cout << "Left stick forward = speed; right stick = yaw while walking. Center = support hold; push = restart.\n";
        std::cout << "Backward/lateral/in-place turns unsupported. [L2 + B] or Ctrl+C -> damping.\n";
    } else if(beyondmimic) {
        std::cout << "Wait 3 seconds, then hold [L2] for 2 seconds and press [Right] to start BeyondMimic.\n";
        std::cout << "[L2 + B] or Ctrl+C -> damping; keep fall-arrest support attached.\n";
    } else {
        std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    }

    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    while (!stop_requested)
    {
        sleep(1);
    }

    fsm->request_shutdown();
    for(int i=0;i<100 && !fsm->shutdown_complete();++i) usleep(10000);
    if(!fsm->shutdown_complete()) {
        spdlog::critical("Controller shutdown timeout; check robot damping and fall-arrest support");
        std::_Exit(1);
    }
    
    return 0;
}

