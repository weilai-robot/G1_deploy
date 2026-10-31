// Hardware-free protocol for the SAME planner, observation and limits used by the FSM.
#include "beyondmimic/cyclic.h"
#include <iostream>
#include <iomanip>
using namespace beyondmimic;
template<class T> void number(std::istream& in,T& value) {
    if(!(in>>value)) throw std::runtime_error("Truncated numeric probe input");
}
Config fixture(const char* path) {
    std::ifstream f(path);if(!f) throw std::runtime_error("Cannot open numeric config fixture");
    Config c;number(f,c.dt);
    for(auto& v:c.sdk_indices) number(f,v);
    for(auto* values:{&c.default_q,&c.kp,&c.kd,&c.action_scale,&c.lower,&c.upper,&c.effort})
        for(auto& v:*values) number(f,v);
    for(auto* chain:{&c.torso,&c.left_foot,&c.right_foot}) {
        int count;number(f,count);if(count<1 || count>29) throw std::runtime_error("Invalid chain length");
        for(int i=0;i<count;++i) {
            Segment s;number(f,s.motor_index);
            for(auto* a:{&s.xyz,&s.rpy,&s.axis}) for(int k=0;k<3;++k) number(f,(*a)[k]);
            chain->push_back(s);
        }
    }
    c.validate();return c;
}
Snapshot snapshot(std::istream& in) {
    Snapshot s;for(auto* a:{&s.q,&s.dq}) for(float& v:*a) number(in,v);
    float w,x,y,z;number(in,w);number(in,x);number(in,y);number(in,z);
    s.orientation=Quat(w,x,y,z);
    for(int k=0;k<3;++k) number(in,s.angular_velocity[k]);
    validate_snapshot(s);s.orientation.normalize();return s;
}
void reference(const CyclicGait& gait) {
    const auto ref=gait.reference();
    std::cout<<gait.phase()<<" "<<gait.rate()<<" "<<gait.cycles()<<" "<<gait.position().x()<<" "<<gait.position().y()<<" "<<gait.yaw_angle();
    for(float v:gait.effective()) std::cout<<" "<<v;
    for(const auto* a:{&ref.q,&ref.dq}) for(float v:*a) std::cout<<" "<<v;
    for(const auto& p:ref.positions) for(int k=0;k<3;++k) std::cout<<" "<<p[k];
    for(const auto& q:ref.orientations) std::cout<<" "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z();
    std::cout<<std::endl;
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: probe gait.bin config.txt initial_support_index");
        CyclicGait gait(argv[1]);gait.reset(std::stoul(argv[3]));Core core(fixture(argv[2]));
        Snapshot current;bool initialised=false;std::cout<<std::setprecision(9);
        std::string command;
        while(std::cin>>command) {
            if(command=="Q") break;
            if(command=="C") {
                std::array<float,3> cmd;for(float& v:cmd) number(std::cin,v);
                int enabled;number(std::cin,enabled);gait.request(cmd,enabled!=0);reference(gait);
            } else if(command=="N") {gait.step();reference(gait);}
            else if(command=="T") reference(gait);
            else if(command=="I") {
                current=snapshot(std::cin);core.reset(current,gait.reference());initialised=true;std::cout<<"OK"<<std::endl;
            } else if(command=="O") {
                if(!initialised) throw std::runtime_error("Observe before reset");
                int truth;number(std::cin,truth);current=snapshot(std::cin);Vec p,v;
                if(truth) {for(int k=0;k<3;++k) number(std::cin,p[k]);for(int k=0;k<3;++k) number(std::cin,v[k]);}
                auto obs=core.observe(current,gait.reference(),true,truth?&p:nullptr,truth?&v:nullptr);
                for(float x:obs) std::cout<<x<<" ";
                for(int k=0;k<3;++k) std::cout<<core.position()[k]<<" ";
                for(int k=0;k<3;++k) std::cout<<core.velocity_b()[k]<<" ";
                std::cout<<std::endl;
            } else if(command=="A") {
                if(!initialised) throw std::runtime_error("Action before reset");
                Joints a;for(float& x:a) number(std::cin,x);
                for(float x:core.targets(a,current,8.0f)) std::cout<<x<<" ";
                std::cout<<core.clipped()<<std::endl;
            } else if(command=="S") {
                float ly,lx,rx;number(std::cin,ly);number(std::cin,lx);number(std::cin,rx);
                for(float x:walking_sticks(ly,lx,rx,gait,.1f,.5f)) std::cout<<x<<" ";
                std::cout<<std::endl;
            } else throw std::runtime_error("Unknown probe command");
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
