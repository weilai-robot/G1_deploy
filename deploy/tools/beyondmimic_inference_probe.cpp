// Binary float32 batches through the actual live Runner; no DDS/robot input.
#include "beyondmimic/package.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=4 && argc!=5) throw std::runtime_error("usage: inference_probe policy.onnx observations.f32 actions.f32");
        const int dim=argc==5?std::stoi(argv[4]):beyondmimic::O;
        beyondmimic::Runner runner(argv[1],dim);
        std::ifstream input(argv[2],std::ios::binary);
        std::ofstream output(argv[3],std::ios::binary);
        if(!input || !output) throw std::runtime_error("Cannot open parity fixture/output");
        std::array<float,beyondmimic::O> obs{};std::size_t count=0;
        const auto record_size=dim*sizeof(float);
        while(input.read(reinterpret_cast<char*>(obs.data()),record_size) || input.gcount()) {
            if(input.gcount()!=static_cast<std::streamsize>(record_size)) throw std::runtime_error("Truncated observation record");
            for(float v:obs) if(!std::isfinite(v)) throw std::runtime_error("Non-finite parity input");
            beyondmimic::Joints action;
            if(dim==beyondmimic::P) {
                std::array<float,beyondmimic::P> p;std::copy_n(obs.begin(),beyondmimic::P,p.begin());action=runner.act(p);
            } else action=runner.act(obs);
            for(float v:action) if(!std::isfinite(v)) throw std::runtime_error("Non-finite ONNX output");
            output.write(reinterpret_cast<char*>(action.data()),sizeof(action));++count;
        }
        if(!count || !output || !input.eof()) throw std::runtime_error("Empty/failed parity I/O");
        std::cout<<count<<" samples\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
