#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace beyondmimic {
struct PlaybackRange {
    std::size_t first, end; // end is exclusive
};

// Full playback uses the table's exact frame count, avoiding duration-to-frame
// rounding and hard-coded durations when a reference motion is replaced.
inline PlaybackRange playback_range(std::size_t frames, float dt,
                                    std::size_t first=0, bool full=true, float duration=10) {
    if(frames<2 || first>=frames || !std::isfinite(dt) || dt<=0)
        throw std::runtime_error("Invalid playback reference/range");
    if(full) return {first,frames};
    if(!std::isfinite(duration) || duration<=0 || duration>300)
        throw std::runtime_error("Invalid playback segment duration");
    const double requested=std::floor(static_cast<double>(duration)/dt);
    const auto available=frames-first;
    const auto steps=requested>=available ? available : static_cast<std::size_t>(requested);
    if(steps==0) throw std::runtime_error("Empty playback segment");
    return {first,first+steps};
}

// Starting directly from FixStand checks the standing pose, not the reference
// first frame. Preparation/legacy playback can still check the reference pose.
template<std::size_t N>
float start_posture_error(const std::array<float,N>& actual_sdk,
                          const std::array<float,N>& standing_sdk,
                          const std::array<float,N>& reference_policy,
                          const std::array<int,N>& sdk_indices, bool from_fixstand) {
    float error=0;
    for(std::size_t i=0;i<N;++i) {
        const int sdk=sdk_indices[i];
        if(sdk<0 || sdk>=static_cast<int>(N)) throw std::runtime_error("Invalid start posture SDK index");
        const float expected=from_fixstand ? standing_sdk[sdk] : reference_policy[i];
        if(!std::isfinite(actual_sdk[sdk]) || !std::isfinite(expected))
            throw std::runtime_error("Invalid start posture feedback");
        error=std::max(error,std::abs(actual_sdk[sdk]-expected));
    }
    return error;
}
} // namespace beyondmimic
