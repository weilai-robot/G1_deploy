#pragma once

// Shared command/reference logic for live control and hardware-free replay.
// Source contract: the archived VelocityGaitReference / CyclicSupportPhaseClock.
#include "beyondmimic/core.h"
#include <limits>

namespace beyondmimic {
class CyclicGait {
public:
    explicit CyclicGait(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open cyclic gait: " + path);
        const auto size=f.tellg(); f.seekg(0);
        char magic[8]; f.read(magic,8);
        std::uint32_t joints,bodies,support_count;
        read(f,frames_); read(f,joints); read(f,bodies); read(f,support_count);
        read(f,dt_); read(f,fps_); read(f,nominal_); read(f,acceleration_);
        for(auto& axis:limits_) {read(f,axis[0]);read(f,axis[1]);}
        const std::uint16_t endian=1;
        if(*reinterpret_cast<const char*>(&endian)!=1 || std::memcmp(magic,"BMGCYC1\0",8) ||
           joints!=J || bodies!=B || frames_<2 || frames_>1000000 ||
           support_count<2 || support_count>frames_ ||
           !finite(dt_) || std::abs(dt_-.02f)>1e-6f || !finite(fps_) ||
           std::abs(fps_*dt_-1)>1e-5f || !finite(nominal_) || nominal_<=0 ||
           !finite(acceleration_) || acceleration_<=0 ||
           size!=static_cast<std::streamoff>(64+support_count*4+std::uint64_t(frames_)*RecordSize*4))
            throw std::runtime_error("Invalid cyclic gait header/size");
        for(const auto& a:limits_) if(!finite(a[0]) || !finite(a[1]) || a[0]>a[1])
            throw std::runtime_error("Invalid trained command limits");
        if(limits_[0][0]!=0 || limits_[0][1]<=0 || limits_[1][0]!=0 || limits_[1][1]!=0 ||
           limits_[2][0]>0 || limits_[2][1]<0)
            throw std::runtime_error("Unsupported cyclic gait command scope");
        supports_.resize(support_count);
        for(auto& s:supports_) read(f,s);
        if(!std::is_sorted(supports_.begin(),supports_.end()) || supports_.back()>=frames_ ||
           std::adjacent_find(supports_.begin(),supports_.end())!=supports_.end())
            throw std::runtime_error("Invalid cyclic support phases");
        data_.resize(std::size_t(frames_)*RecordSize);
        f.read(reinterpret_cast<char*>(data_.data()),data_.size()*sizeof(float));
        if(!f) throw std::runtime_error("Truncated cyclic gait");
        for(float v:data_) if(!finite(v)) throw std::runtime_error("Non-finite gait tensor");
        for(std::size_t i=0;i<frames_;++i) for(int b=0;b<B;++b) {
            const float* q=&data_[i*RecordSize+2*J+3*B+4*b];
            if(std::abs(Quat(q[0],q[1],q[2],q[3]).norm()-1)>.01f)
                throw std::runtime_error("Invalid gait quaternion");
        }
        reset(0);
    }
    void reset(std::size_t support_index) {
        if(support_index>=supports_.size()) throw std::runtime_error("Invalid initial support");
        phase_=float(supports_[support_index]); cycles_=0; rate_=0; was_go_=false;
        remaining_=0; target_phase_=phase_; target_cycle_=0;
        root_.setZero(); yaw_=0; raw_.fill(0); bounded_.fill(0); effective_.fill(0); unit_.fill(0);
        saturated_=false;
    }
    void request(const std::array<float,3>& raw,bool enabled=true) {
        std::array<float,3> bounded{}; bool saturated=false;
        for(int i=0;i<3;++i) {
            if(!finite(raw[i])) throw std::runtime_error("Non-finite velocity command");
            if(enabled && limits_[i][0]==limits_[i][1] && raw[i]!=limits_[i][0])
                throw std::runtime_error("Requested unsupported command axis");
            const float value=std::clamp(raw[i],limits_[i][0],limits_[i][1]);
            saturated|=value!=raw[i]; bounded[i]=enabled?value:0;
        }
        if(enabled && std::hypot(bounded[0],bounded[1])==0 && bounded[2]!=0)
            throw std::runtime_error("In-place turning is not supported by the trained gait");
        raw_=raw; bounded_=bounded; saturated_=saturated;
    }
    void step() {
        const float target=std::hypot(bounded_[0],bounded_[1])/nominal_;
        const bool go=target>0;
        if(go) for(int i=0;i<3;++i) unit_[i]=bounded_[i]/target;
        if(was_go_ && !go) {
            const float earliest=phase_+rate_*rate_*fps_/(2*acceleration_)+rate_*fps_*dt_;
            auto offset=static_cast<std::int64_t>(std::floor(earliest/frames_));
            const float local=earliest-float(offset*frames_);
            auto it=std::lower_bound(supports_.begin(),supports_.end(),local);
            if(it==supports_.end()) {++offset;it=supports_.begin();}
            target_phase_=float(*it); target_cycle_=cycles_+offset;
            remaining_=float(offset*frames_)+target_phase_-phase_;
        }
        const float old_phase=phase_; const auto old_cycle=cycles_;
        const float advancing=rate_+std::clamp(target-rate_,-acceleration_*dt_,acceleration_*dt_);
        const float cap=std::sqrt(2*acceleration_*std::max(0.0f,remaining_)/fps_);
        float next_rate=go?advancing:std::min(rate_,cap);
        float advance=(rate_+next_rate)*.5f*dt_*fps_;
        if(!go) advance=std::min(advance,remaining_);
        const float next=phase_+advance;
        const auto wraps=static_cast<std::int64_t>(std::floor(next/frames_));
        phase_=next-float(wraps*frames_); cycles_+=wraps;
        if(!go) remaining_=std::max(0.0f,remaining_-advance);
        if(!go && remaining_<=1e-5f) {phase_=target_phase_;cycles_=target_cycle_;next_rate=0;}
        const float actual_advance=float((cycles_-old_cycle)*frames_)+phase_-old_phase;
        const float mean_rate=actual_advance/(fps_*dt_);
        const float angle=unit_[2]*mean_rate*dt_;
        const float middle=yaw_+angle*.5f;
        constexpr float pi=3.14159265358979323846f;
        const float sinc_arg=angle/(2*pi), t=pi*sinc_arg;
        const float scale=dt_*(t==0?1:std::sin(t)/t);
        const float vx=unit_[0]*mean_rate, vy=unit_[1]*mean_rate;
        root_.x()+=scale*(std::cos(middle)*vx-std::sin(middle)*vy);
        root_.y()+=scale*(std::sin(middle)*vx+std::cos(middle)*vy);
        yaw_+=angle;rate_=next_rate;was_go_=go;
        for(int i=0;i<3;++i) effective_[i]=unit_[i]*rate_;
    }
    Reference reference() const {
        const auto lo=static_cast<std::size_t>(phase_), hi=(lo+1)%frames_;
        const float a=phase_-float(lo);
        const float* left=&data_[lo*RecordSize],*right=&data_[hi*RecordSize];
        auto lerp=[a](float l,float r){return l+(r-l)*a;};
        Reference out;
        for(int i=0;i<J;++i) {out.q[i]=lerp(left[i],right[i]);out.dq[i]=lerp(left[J+i],right[J+i])*rate_;}
        const Quat heading(Eigen::AngleAxisf(yaw_,Vec::UnitZ()));
        for(int b=0;b<B;++b) {
            const int p=2*J+3*b,q=2*J+3*B+4*b;
            out.positions[b]=heading*Vec(lerp(left[p],right[p]),lerp(left[p+1],right[p+1]),lerp(left[p+2],right[p+2]));
            out.positions[b].x()+=root_.x();out.positions[b].y()+=root_.y();
            Quat l(left[q],left[q+1],left[q+2],left[q+3]),r(right[q],right[q+1],right[q+2],right[q+3]);
            if(l.coeffs().dot(r.coeffs())<0) r.coeffs()=-r.coeffs();
            Quat v;v.coeffs()=l.coeffs()+a*(r.coeffs()-l.coeffs());v.normalize();
            out.orientations[b]=heading*v;
        }
        return out;
    }
    float phase() const{return phase_;} float rate() const{return rate_;}
    std::int64_t cycles() const{return cycles_;} float yaw_angle() const{return yaw_;}
    float dt() const{return dt_;} float nominal_speed() const{return nominal_;}
    const Eigen::Vector2f& position() const{return root_;}
    const std::array<float,3>& raw() const{return raw_;}
    const std::array<float,3>& bounded() const{return bounded_;}
    const std::array<float,3>& effective() const{return effective_;}
    const std::array<std::array<float,2>,3>& limits() const{return limits_;}
    bool saturated() const{return saturated_;}
private:
    template<class T> static void read(std::ifstream& f,T& v) {f.read(reinterpret_cast<char*>(&v),sizeof(v));if(!f) throw std::runtime_error("Truncated gait header");}
    static bool finite(float v){return std::isfinite(v);}
    std::uint32_t frames_=0;float dt_=0,fps_=0,nominal_=0,acceleration_=0;
    std::array<std::array<float,2>,3> limits_{};
    std::vector<std::uint32_t> supports_;std::vector<float> data_;
    float phase_=0,rate_=0,remaining_=0,target_phase_=0,yaw_=0;
    std::int64_t cycles_=0,target_cycle_=0;bool was_go_=false,saturated_=false;
    Eigen::Vector2f root_=Eigen::Vector2f::Zero();
    std::array<float,3> raw_{},bounded_{},effective_{},unit_{};
};

// Maps only the trained joystick axes. Unsupported inputs never request a turn in place.
inline std::array<float,3> walking_sticks(float ly,float lx,float rx,const CyclicGait& gait,float deadzone,float minimum_speed_scale=.5f) {
    if(!std::isfinite(ly)||!std::isfinite(lx)||!std::isfinite(rx) || std::abs(ly)>1.01f ||
       std::abs(lx)>1.01f || std::abs(rx)>1.01f || !(deadzone>=0 && deadzone<1) ||
       !(minimum_speed_scale>0 && minimum_speed_scale<=1)) throw std::runtime_error("Invalid joystick input/settings");
    if(ly<=deadzone) return {0,0,0};
    const float forward=std::clamp((ly-deadzone)/(1-deadzone),0.0f,1.0f);
    const float speed=gait.nominal_speed()*(minimum_speed_scale+(1-minimum_speed_scale)*forward);
    const float turn=std::abs(rx)<=deadzone?0:std::copysign((std::abs(rx)-deadzone)/(1-deadzone),-rx);
    const float yaw_rate=turn>=0?turn*gait.limits()[2][1]:-turn*gait.limits()[2][0];
    return {std::min(speed,gait.limits()[0][1]),0,yaw_rate};
}
} // namespace beyondmimic
