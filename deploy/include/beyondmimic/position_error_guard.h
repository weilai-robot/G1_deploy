#pragma once

#include <cmath>
#include <stdexcept>

namespace beyondmimic {
// Returns an exceedance for monitoring even when the distance stop is disabled.
// Invalid observations are always rejected, independently of that switch.
inline bool check_anchor_position_error(float error, float limit, bool enforce=true) {
    if(!std::isfinite(error) || error<0 || !std::isfinite(limit) || limit<=0)
        throw std::runtime_error("Invalid torso/reference position error");
    const bool exceeded=error>limit;
    if(enforce && exceeded)
        throw std::runtime_error("Estimated torso/reference position error limit exceeded");
    return exceeded;
}
} // namespace beyondmimic
