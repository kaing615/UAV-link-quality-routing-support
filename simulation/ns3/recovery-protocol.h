#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace recovery
{
struct Action { bool quit=false; std::vector<double> values; };
inline Action ReadAction(std::istream &input, uint32_t expectedStep, size_t count)
{
    std::string line;
    if (!std::getline(input,line)) { throw std::runtime_error("Controller closed input without UAVQUIT"); }
    if (line == "UAVQUIT" || line == "UAVQUIT\r") { return {true,{}}; }
    std::istringstream parser(line);
    std::string prefix;
    uint32_t version=0, step=0;
    if (!(parser>>prefix>>version>>step) || prefix!="UAVACT" || version!=1 || step!=expectedStep)
        throw std::runtime_error("Invalid action protocol version or step");
    Action action;
    for (size_t i=0; i<count; ++i)
    {
        double value;
        if (!(parser>>value) || !std::isfinite(value) || value < -1.000001 || value > 1.000001)
            throw std::runtime_error("Action must contain finite normalized values");
        action.values.push_back(std::clamp(value,-1.0,1.0));
    }
    std::string extra;
    if (parser>>extra) { throw std::runtime_error("Extra action values"); }
    return action;
}
} // namespace recovery
