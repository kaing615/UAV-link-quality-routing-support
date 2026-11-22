#pragma once

// Version-2 controller line for --routeControl=external (v1 stays in recovery-protocol.h):
//   UAVACT 2 <step> <3*relays normalized velocities> ROUTE <m> <node_1> ... <node_m>
// m == 0 releases any override (live OLSR decides). Otherwise the nodes are a
// simple path source..destination that unicast data to the destination follows.

#include "recovery-protocol.h"

#include <set>

namespace recovery
{
struct RoutedAction { bool quit=false; std::vector<double> values; std::vector<uint32_t> route; };

inline void ValidateRoute(const std::vector<uint32_t> &route, uint32_t numNodes, uint32_t source, uint32_t destination)
{
    if (route.empty()) return;
    if (route.size()<2 || route.front()!=source || route.back()!=destination)
        throw std::runtime_error("Route must run from source to destination");
    std::set<uint32_t> seen;
    for (uint32_t node : route)
        if (node>=numNodes || !seen.insert(node).second) throw std::runtime_error("Route must be a simple path of known nodes");
}

inline RoutedAction ReadRoutedAction(std::istream &input, uint32_t expectedStep, size_t count,
                                     uint32_t numNodes, uint32_t source, uint32_t destination)
{
    std::string line;
    if (!std::getline(input,line)) { throw std::runtime_error("Controller closed input without UAVQUIT"); }
    if (line == "UAVQUIT" || line == "UAVQUIT\r") { return {true,{},{}}; }
    std::istringstream parser(line);
    std::string prefix, marker;
    uint32_t version=0, step=0;
    if (!(parser>>prefix>>version>>step) || prefix!="UAVACT" || version!=2 || step!=expectedStep)
        throw std::runtime_error("Invalid routed action protocol version or step");
    RoutedAction action;
    for (size_t i=0; i<count; ++i)
    {
        double value;
        if (!(parser>>value) || !std::isfinite(value) || value < -1.000001 || value > 1.000001)
            throw std::runtime_error("Action must contain finite normalized values");
        action.values.push_back(std::clamp(value,-1.0,1.0));
    }
    long long m=-1;
    if (!(parser>>marker>>m) || marker!="ROUTE" || m<0 || m>static_cast<long long>(numNodes))
        throw std::runtime_error("Routed action requires ROUTE <m>");
    for (long long i=0; i<m; ++i)
    {
        long long node=-1;
        if (!(parser>>node) || node<0) throw std::runtime_error("Invalid route node");
        action.route.push_back(static_cast<uint32_t>(node));
    }
    std::string extra;
    if (parser>>extra) { throw std::runtime_error("Extra routed action values"); }
    ValidateRoute(action.route,numNodes,source,destination);
    return action;
}
} // namespace recovery
