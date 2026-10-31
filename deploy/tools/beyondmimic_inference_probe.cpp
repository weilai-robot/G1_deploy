// Binary float32 batches through the actual live Runner; no DDS/robot input.
#include "beyondmimic/package.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: inference_probe policy.onnx observations.f32 actions.f32");
        beyondmimic::Runner runner(argv[1]);
        std::ifstream input(argv[2],std::ios::binary);
        std::ofstream output(argv[3],std::ios::binary);
        if(!input || !output) throw std::runtime_error("Cannot open parity fixture/output");
        std::array<float,beyondmimic::O> obs{};std::size_t count=0;
        while(input.read(reinterpret_cast<char*>(obs.data()),sizeof(obs)) || input.gcount()) {
            if(input.gcount()!=sizeof(obs)) throw std::runtime_error("Truncated observation record");
            for(float v:obs) if(!std::isfinite(v)) throw std::runtime_error("Non-finite parity input");
            auto action=runner.act(obs);
            for(float v:action) if(!std::isfinite(v)) throw std::runtime_error("Non-finite ONNX output");
            output.write(reinterpret_cast<char*>(action.data()),sizeof(action));++count;
        }
        if(!count || !output || !input.eof()) throw std::runtime_error("Empty/failed parity I/O");
        std::cout<<count<<" samples\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
