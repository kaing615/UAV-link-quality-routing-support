/*
 * UAV link-quality dataset generator on ns-3.
 *
 * Replaces the formula-based Python simulator (simulation/main.py) with a real
 * 802.11 ad-hoc stack + real OLSR: RSSI is sniffed from the PHY, delay /
 * packet_loss are measured from broadcast UDP probe packets, and connectivity
 * emerges from actual packet reception instead of a distance cutoff.
 *
 * Output schema is identical to the Python simulator so the entire existing
 * pipeline (preprocessing -> graph dataset -> training) works unchanged:
 *
 *   nodes.csv       time,node_id,x,y,z,vx,vy,vz,speed,degree
 *   edges.csv       time,src,dst,distance,connected,relative_speed,rssi,snr,
 *                   delay,packet_loss,throughput,p_stable,weight
 *   traffic_log.csv time,source,destination,reachable,route_path,num_edges,
 *                   routing_protocol,olsr_mpr_nodes,olsr_avg_rt_size
 *   scenario.json   run configuration
 *
 * Feature semantics vs the Python simulator:
 *   rssi        : mean sniffed signal (dBm) over the last 1 s window; falls
 *                 back to the deterministic log-distance value when no packet
 *                 was received (same propagation model -> same scale)
 *   snr         : rssi - noise_floor_dbm (fixed floor for scale parity)
 *   delay       : mean measured one-hop probe delay (ms); disconnected pairs
 *                 get base_delay + propagation + disconnected_penalty
 *   packet_loss : 1 - received/expected probes (both directions combined)
 *   connected   : probe delivery ratio >= 0.5 in the last window
 *   throughput  : derived from measured snr/loss/load with the same formula
 *                 as simulation/metrics.py (capacity proxy, not probe goodput)
 *   p_stable    : same scoring formula as simulation/metrics.py applied to
 *                 the measured snr/loss/delay; weight = 1 - p_stable
 *
 * Snapshots are taken every 1 s after a warm-up period (OLSR convergence);
 * snapshot t aggregates the probe window [t, t+1) relative to warm-up end.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavOlsrDataset");

static uint32_t g_numUavs = 6;
static uint32_t g_timeSteps = 145;
static uint32_t g_seed = 42;
static std::string g_mobility = "gauss-markov"; // or "random-waypoint"
static double g_xMax = 500.0;
static double g_yMax = 500.0;
static double g_zMin = 50.0;
static double g_zMax = 150.0;
static double g_commRange = 243.0;
static double g_txPowerDbm = 20.0;
static double g_refLossDb = 40.0;
static double g_pathLossExp = 2.2;
static double g_noiseFloorDbm = -90.0;
static double g_baseDelayMs = 2.0;
static double g_disconnectedDelayMs = 50.0;
static double g_maxThroughputMbps = 100.0;
static double g_gmAlpha = 0.85;
static double g_rwpSpeedMin = 3.0;
static double g_rwpSpeedMax = 8.0;
static double g_warmup = 10.0;
static uint32_t g_sourceId = 0;
static uint32_t g_destId = 4;
static std::string g_outputDir = ".";
static bool g_enableAnim = true;
static bool g_enableDataFlow = false;
static std::string g_routePlanPath;
static std::string g_routingStrategy = "olsr";
static std::string g_predictionTarget = "none";
static uint32_t g_predictionHorizon = 1;
static std::string g_costMode = "none";
static double g_appRateKbps = 256.0;
static uint32_t g_appPacketSize = 512;

static constexpr double PROBE_INTERVAL = 0.05;
static constexpr uint16_t PROBE_PORT = 9999;
static constexpr uint16_t DATA_PORT = 9000;

// ---------------------------------------------------------------------------
struct LinkWindow
{
    uint32_t rxCount = 0;
    double delaySumMs = 0.0;
    uint32_t rssiCount = 0;
    double rssiSumDbm = 0.0;
};

static NodeContainer g_nodes;
static Ipv4InterfaceContainer g_ifaces;
static std::vector<Ptr<Socket>> g_txSockets;
static std::map<Mac48Address, uint32_t> g_macToNode;
static std::map<Ipv4Address, uint32_t> g_ipToNode;
// key: ordered pair (txNode, rxNode)
static std::map<std::pair<uint32_t, uint32_t>, LinkWindow> g_window;
// degree of each node at the previous snapshot (load proxy for throughput)
static std::vector<uint32_t> g_prevDegree;

static std::ofstream g_nodesCsv;
static std::ofstream g_edgesCsv;
static std::ofstream g_trafficCsv;

struct RoutePlanEntry
{
    bool found = false;
    std::vector<uint32_t> path;
};

static std::map<uint32_t, RoutePlanEntry> g_routePlan;
static std::vector<uint32_t> g_currentPath;
static uint32_t g_planSteps = 0;
static uint32_t g_planFound = 0;
static uint32_t g_routeChanges = 0;

static double
Clamp01(double x)
{
    return std::max(0.0, std::min(1.0, x));
}

static double
FormulaRssi(double distance)
{
    double d = std::max(distance, 1.0);
    return g_txPowerDbm - (g_refLossDb + 10.0 * g_pathLossExp * std::log10(d));
}

static double
EstimateThroughput(double snr, double packetLoss, bool connected, double loadFactor)
{
    if (!connected)
    {
        return 0.0;
    }
    double snrEff = Clamp01(snr / 30.0);
    double tp = g_maxThroughputMbps * snrEff * (1.0 - packetLoss) * (1.0 - 0.35 * loadFactor);
    return std::max(tp, 0.0);
}

static double
EstimatePStable(double snr, double packetLoss, double delayMs, bool connected)
{
    if (!connected)
    {
        return 0.0;
    }
    double snrScore = Clamp01((snr - 5.0) / 20.0);
    double lossScore = Clamp01(1.0 - packetLoss);
    double delayScore = Clamp01(1.0 - delayMs / 50.0);
    return Clamp01(0.45 * snrScore + 0.35 * lossScore + 0.20 * delayScore);
}

static void
SendProbe(uint32_t nodeId)
{
    uint8_t buf[12];
    uint64_t now = Simulator::Now().GetNanoSeconds();
    uint32_t id = nodeId;
    std::memcpy(buf, &now, 8);
    std::memcpy(buf + 8, &id, 4);
    Ptr<Packet> p = Create<Packet>(buf, 12);
    g_txSockets[nodeId]->Send(p);
    Simulator::Schedule(Seconds(PROBE_INTERVAL), &SendProbe, nodeId);
}

static void
ReceiveProbe(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        if (packet->GetSize() < 12)
        {
            continue;
        }
        uint8_t buf[12];
        packet->CopyData(buf, 12);
        uint64_t txNs;
        uint32_t srcId;
        std::memcpy(&txNs, buf, 8);
        std::memcpy(&srcId, buf + 8, 4);

        uint32_t rxId = socket->GetNode()->GetId();
        if (srcId == rxId || srcId >= g_numUavs)
        {
            continue;
        }
        double delayMs = (Simulator::Now().GetNanoSeconds() - txNs) / 1e6;
        LinkWindow &w = g_window[{srcId, rxId}];
        w.rxCount += 1;
        w.delaySumMs += delayMs;
    }
}

static void
MonitorSniffRx(std::string context,
               Ptr<const Packet> packet,
               uint16_t,
               WifiTxVector,
               MpduInfo,
               SignalNoiseDbm signalNoise,
               uint16_t)
{
    uint32_t rxId = std::stoul(context.substr(10, context.find('/', 10) - 10));

    WifiMacHeader hdr;
    if (!packet->PeekHeader(hdr))
    {
        return;
    }
    auto it = g_macToNode.find(hdr.GetAddr2());
    if (it == g_macToNode.end() || it->second == rxId)
    {
        return;
    }
    LinkWindow &w = g_window[{it->second, rxId}];
    w.rssiCount += 1;
    w.rssiSumDbm += signalNoise.signal;
}

static Ptr<olsr::RoutingProtocol>
GetOlsr(Ptr<Node> node)
{
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ptr<Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol();
    Ptr<Ipv4ListRouting> list = DynamicCast<Ipv4ListRouting>(rp);
    if (!list)
    {
        return DynamicCast<olsr::RoutingProtocol>(rp);
    }
    for (uint32_t i = 0; i < list->GetNRoutingProtocols(); ++i)
    {
        int16_t prio;
        Ptr<olsr::RoutingProtocol> o =
            DynamicCast<olsr::RoutingProtocol>(list->GetRoutingProtocol(i, prio));
        if (o)
        {
            return o;
        }
    }
    return nullptr;
}

static bool
WalkRoute(uint32_t src, uint32_t dst, std::vector<uint32_t> &path)
{
    Ipv4Address dstAddr =
        g_nodes.Get(dst)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    path.clear();
    path.push_back(src);
    uint32_t current = src;
    for (uint32_t hop = 0; hop < g_numUavs; ++hop)
    {
        if (current == dst)
        {
            return true;
        }
        Ptr<olsr::RoutingProtocol> o = GetOlsr(g_nodes.Get(current));
        if (!o)
        {
            return false;
        }
        bool found = false;
        for (const auto &entry : o->GetRoutingTableEntries())
        {
            if (entry.destAddr == dstAddr)
            {
                auto it = g_ipToNode.find(entry.nextAddr);
                if (it == g_ipToNode.end())
                {
                    return false;
                }
                current = it->second;
                path.push_back(current);
                found = true;
                break;
            }
        }
        if (!found)
        {
            return false;
        }
    }
    return current == dst;
}

static std::vector<uint32_t>
ParseRoutePath(const std::string &value)
{
    std::vector<uint32_t> path;
    if (value.empty())
    {
        return path;
    }
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, '>'))
    {
        if (!token.empty() && token.back() == '-')
        {
            token.pop_back();
        }
        if (!token.empty())
        {
            path.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }
    return path;
}

static void
LoadRoutePlan(const std::string &path)
{
    std::ifstream input(path);
    NS_ABORT_MSG_IF(!input, "Cannot open route plan: " << path);
    std::string line;
    std::getline(input, line); // header
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream row(line);
        std::string field;
        while (std::getline(row, field, ','))
        {
            fields.push_back(field);
        }
        NS_ABORT_MSG_IF(fields.size() < 5, "Malformed route-plan row: " << line);
        uint32_t step = static_cast<uint32_t>(std::stoul(fields[0]));
        uint32_t source = static_cast<uint32_t>(std::stoul(fields[1]));
        uint32_t destination = static_cast<uint32_t>(std::stoul(fields[2]));
        bool found = std::stoul(fields[3]) == 1;
        std::vector<uint32_t> nodes = ParseRoutePath(fields[4]);
        NS_ABORT_MSG_IF(source != g_sourceId || destination != g_destId,
                        "Route-plan source/destination does not match CLI arguments");
        NS_ABORT_MSG_IF(g_routePlan.count(step) != 0, "Duplicate route-plan time " << step);
        if (found)
        {
            NS_ABORT_MSG_IF(nodes.size() < 2 || nodes.front() != source || nodes.back() != destination,
                            "Invalid route path at time " << step);
            NS_ABORT_MSG_IF(*std::max_element(nodes.begin(), nodes.end()) >= g_numUavs,
                            "Route path contains an unknown node at time " << step);
        }
        g_routePlan[step] = {found, nodes};
    }
    NS_ABORT_MSG_IF(g_routePlan.empty(), "Route plan contains no usable rows: " << path);
}

static Ptr<Ipv4StaticRouting>
GetStaticRouting(uint32_t nodeId)
{
    Ipv4StaticRoutingHelper helper;
    return helper.GetStaticRouting(g_nodes.Get(nodeId)->GetObject<Ipv4>());
}

static void
RemovePlannedRoutes()
{
    Ipv4Address destination = g_ifaces.GetAddress(g_destId);
    for (uint32_t nodeId = 0; nodeId < g_numUavs; ++nodeId)
    {
        Ptr<Ipv4StaticRouting> routing = GetStaticRouting(nodeId);
        for (uint32_t index = routing->GetNRoutes(); index > 0; --index)
        {
            if (routing->GetRoute(index - 1).GetDest() == destination)
            {
                routing->RemoveRoute(index - 1);
            }
        }
    }
}

static void
ApplyRoutePlan(uint32_t step)
{
    auto plan = g_routePlan.find(step);
    if (plan == g_routePlan.end())
    {
        return;
    }
    const RoutePlanEntry &entry = plan->second;
    if (g_planSteps > 0 && entry.path != g_currentPath)
    {
        g_routeChanges += 1;
    }
    g_planSteps += 1;
    g_planFound += entry.found ? 1 : 0;
    g_currentPath = entry.path;
    RemovePlannedRoutes();
    if (!entry.found)
    {
        Ptr<Ipv4> sourceIpv4 = g_nodes.Get(g_sourceId)->GetObject<Ipv4>();
        int32_t sourceInterface =
            sourceIpv4->GetInterfaceForDevice(g_nodes.Get(g_sourceId)->GetDevice(0));
        NS_ABORT_MSG_IF(sourceInterface < 0, "No IPv4 interface on source node");
        GetStaticRouting(g_sourceId)->AddHostRouteTo(
            g_ifaces.GetAddress(g_destId), static_cast<uint32_t>(sourceInterface));
        return;
    }

    Ipv4Address destination = g_ifaces.GetAddress(g_destId);
    for (size_t index = 0; index + 1 < entry.path.size(); ++index)
    {
        uint32_t current = entry.path[index];
        uint32_t next = entry.path[index + 1];
        Ptr<Ipv4> ipv4 = g_nodes.Get(current)->GetObject<Ipv4>();
        int32_t interface = ipv4->GetInterfaceForDevice(g_nodes.Get(current)->GetDevice(0));
        NS_ABORT_MSG_IF(interface < 0, "No IPv4 interface for planned route node " << current);
        GetStaticRouting(current)->AddHostRouteTo(destination,
                                                  g_ifaces.GetAddress(next),
                                                  static_cast<uint32_t>(interface));
    }
}

static void
TakeSnapshot(uint32_t step)
{
    const double expectedPerDir = 1.0 / PROBE_INTERVAL; // probes per window per direction
    const uint32_t n = g_numUavs;

    std::vector<Vector> pos(n);
    std::vector<Vector> vel(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        Ptr<MobilityModel> mm = g_nodes.Get(i)->GetObject<MobilityModel>();
        pos[i] = mm->GetPosition();
        vel[i] = mm->GetVelocity();
    }

    std::vector<uint32_t> degree(n, 0);
    uint32_t numConnectedEdges = 0;

    struct EdgeRow
    {
        uint32_t src;
        uint32_t dst;
        double distance;
        int connected;
        double relSpeed;
        double rssi;
        double snr;
        double delay;
        double loss;
    };

    std::vector<EdgeRow> rows;
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = i + 1; j < n; ++j)
        {
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double rvx = vel[i].x - vel[j].x;
            double rvy = vel[i].y - vel[j].y;
            double rvz = vel[i].z - vel[j].z;
            double relSpeed = std::sqrt(rvx * rvx + rvy * rvy + rvz * rvz);

            const LinkWindow &fwd = g_window[{i, j}];
            const LinkWindow &bwd = g_window[{j, i}];

            uint32_t rxTotal = fwd.rxCount + bwd.rxCount;
            double rxRatio = rxTotal / (2.0 * expectedPerDir);
            bool connected = rxRatio >= 0.5;

            double rssi;
            uint32_t rssiCount = fwd.rssiCount + bwd.rssiCount;
            if (rssiCount > 0)
            {
                rssi = (fwd.rssiSumDbm + bwd.rssiSumDbm) / rssiCount;
            }
            else
            {
                rssi = FormulaRssi(dist);
            }
            double snr = rssi - g_noiseFloorDbm;

            double delayMs;
            if (rxTotal > 0)
            {
                delayMs = (fwd.delaySumMs + bwd.delaySumMs) / rxTotal;
                if (!connected)
                {
                    delayMs += g_disconnectedDelayMs;
                }
            }
            else
            {
                delayMs = g_baseDelayMs + dist / 3.0e8 * 1000.0 + g_disconnectedDelayMs;
            }

            double loss = connected ? Clamp01(1.0 - rxRatio) : 1.0;

            if (connected)
            {
                degree[i] += 1;
                degree[j] += 1;
                numConnectedEdges += 1;
            }
            rows.push_back({i, j, dist, connected ? 1 : 0, relSpeed, rssi, snr, delayMs, loss});
        }
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        double speed =
            std::sqrt(vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
        g_nodesCsv << step << ',' << i << ',' << pos[i].x << ',' << pos[i].y << ',' << pos[i].z
                   << ',' << vel[i].x << ',' << vel[i].y << ',' << vel[i].z << ',' << speed << ','
                   << degree[i] << '\n';
    }

    for (const auto &r : rows)
    {
        double denom = std::max<double>(n - 1, 1);
        double loadFactor =
            ((g_prevDegree[r.src] / denom) + (g_prevDegree[r.dst] / denom)) / 2.0;
        double throughput =
            EstimateThroughput(r.snr, r.loss, r.connected == 1, loadFactor);
        double pStable = EstimatePStable(r.snr, r.loss, r.delay, r.connected == 1);
        double weight = 1.0 - pStable;

        g_edgesCsv << step << ',' << r.src << ',' << r.dst << ',' << r.distance << ','
                   << r.connected << ',' << r.relSpeed << ',' << r.rssi << ',' << r.snr << ','
                   << r.delay << ',' << r.loss << ',' << throughput << ',' << pStable << ','
                   << weight << '\n';
    }
    g_prevDegree = degree;

    std::vector<uint32_t> path;
    bool reachable = false;
    if (g_routingStrategy == "olsr")
    {
        reachable = WalkRoute(g_sourceId, g_destId, path);
        if (g_routePlan.count(step) != 0)
        {
            if (g_planSteps > 0 && path != g_currentPath)
            {
                g_routeChanges += 1;
            }
            g_planSteps += 1;
            g_planFound += reachable ? 1 : 0;
            g_currentPath = path;
        }
    }
    else
    {
        auto plan = g_routePlan.find(step);
        if (plan != g_routePlan.end() && plan->second.found)
        {
            path = plan->second.path;
            reachable = true;
        }
    }
    std::ostringstream pathStr;
    if (reachable)
    {
        for (size_t k = 0; k < path.size(); ++k)
        {
            if (k)
            {
                pathStr << "->";
            }
            pathStr << path[k];
        }
    }
    double rtSizeSum = 0.0;
    if (g_routingStrategy == "olsr")
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            Ptr<olsr::RoutingProtocol> o = GetOlsr(g_nodes.Get(i));
            rtSizeSum += o ? o->GetRoutingTableEntries().size() : 0;
        }
    }
    g_trafficCsv << step << ',' << g_sourceId << ',' << g_destId << ',' << (reachable ? 1 : 0)
                 << ',' << pathStr.str() << ',' << numConnectedEdges << ',' << g_routingStrategy
                 << ",0,"
                 << rtSizeSum / n << '\n';

    g_window.clear();
}

static void
WriteClosedLoopMetrics(Ptr<FlowMonitor> monitor, FlowMonitorHelper &helper)
{
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(helper.GetClassifier());
    Ipv4Address sourceAddress = g_ifaces.GetAddress(g_sourceId);
    Ipv4Address destinationAddress = g_ifaces.GetAddress(g_destId);
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;
    uint64_t lostPackets = 0;
    uint64_t rxBytes = 0;
    double delaySumSeconds = 0.0;
    double firstTxSeconds = std::numeric_limits<double>::infinity();
    double lastRxSeconds = 0.0;

    for (const auto &[flowId, stats] : monitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flowId);
        if (tuple.sourceAddress != sourceAddress || tuple.destinationAddress != destinationAddress ||
            tuple.destinationPort != DATA_PORT)
        {
            continue;
        }
        txPackets += stats.txPackets;
        rxPackets += stats.rxPackets;
        lostPackets += stats.lostPackets;
        rxBytes += stats.rxBytes;
        delaySumSeconds += stats.delaySum.GetSeconds();
        if (stats.txPackets > 0)
        {
            firstTxSeconds = std::min(firstTxSeconds, stats.timeFirstTxPacket.GetSeconds());
        }
        if (stats.rxPackets > 0)
        {
            lastRxSeconds = std::max(lastRxSeconds, stats.timeLastRxPacket.GetSeconds());
        }
    }

    double pdr = txPackets > 0 ? static_cast<double>(rxPackets) / txPackets : 0.0;
    double meanDelayMs = rxPackets > 0 ? delaySumSeconds * 1000.0 / rxPackets : 0.0;
    double duration = std::isfinite(firstTxSeconds) && lastRxSeconds > firstTxSeconds
                          ? lastRxSeconds - firstTxSeconds
                          : 0.0;
    double throughputMbps = duration > 0.0 ? rxBytes * 8.0 / duration / 1.0e6 : 0.0;
    double foundRate = g_planSteps > 0 ? static_cast<double>(g_planFound) / g_planSteps : 0.0;

    std::ofstream output(g_outputDir + "/closed_loop_metrics.csv");
    output << std::fixed << std::setprecision(6)
           << "strategy,target,horizon,cost_mode,seed,source,destination,app_rate_kbps,"
              "packet_size,tx_packets,rx_packets,"
              "lost_packets,pdr,mean_delay_ms,throughput_mbps,route_changes,plan_steps,"
              "route_found_rate\n"
           << g_routingStrategy << ',' << g_predictionTarget << ',' << g_predictionHorizon << ','
           << g_costMode << ',' << g_seed << ',' << g_sourceId << ',' << g_destId << ','
           << g_appRateKbps << ',' << g_appPacketSize << ',' << txPackets << ',' << rxPackets << ','
           << lostPackets << ',' << pdr << ','
           << meanDelayMs << ',' << throughputMbps << ',' << g_routeChanges << ',' << g_planSteps
           << ',' << foundRate << '\n';
}

static void
WriteScenarioJson(const std::string &runName)
{
    std::ofstream f(g_outputDir + "/scenario.json");
    f << std::fixed << std::setprecision(4);
    f << "{\n"
      << "  \"run_name\": \"" << runName << "\",\n"
      << "  \"simulator\": \"ns-3\",\n"
      << "  \"seed\": " << g_seed << ",\n"
      << "  \"mobility_model\": \"" << g_mobility << "\",\n"
      << "  \"num_uavs\": " << g_numUavs << ",\n"
      << "  \"time_steps\": " << g_timeSteps << ",\n"
      << "  \"dt\": 1.0,\n"
      << "  \"warmup_s\": " << g_warmup << ",\n"
      << "  \"x_limit\": [0.0, " << g_xMax << "],\n"
      << "  \"y_limit\": [0.0, " << g_yMax << "],\n"
      << "  \"z_limit\": [" << g_zMin << ", " << g_zMax << "],\n"
      << "  \"comm_range\": " << g_commRange << ",\n"
      << "  \"source_id\": " << g_sourceId << ",\n"
      << "  \"dest_id\": " << g_destId << ",\n"
      << "  \"gauss_markov_alpha\": " << g_gmAlpha << ",\n"
      << "  \"rwp_speed_range\": [" << g_rwpSpeedMin << ", " << g_rwpSpeedMax << "],\n"
      << "  \"tx_power_dbm\": " << g_txPowerDbm << ",\n"
      << "  \"reference_path_loss_db\": " << g_refLossDb << ",\n"
      << "  \"path_loss_exponent\": " << g_pathLossExp << ",\n"
      << "  \"noise_floor_dbm\": " << g_noiseFloorDbm << ",\n"
      << "  \"base_delay_ms\": " << g_baseDelayMs << ",\n"
      << "  \"disconnected_delay_ms\": " << g_disconnectedDelayMs << ",\n"
      << "  \"max_throughput_mbps\": " << g_maxThroughputMbps << ",\n"
      << "  \"probe_interval_s\": " << PROBE_INTERVAL << ",\n"
      << "  \"wifi_standard\": \"802.11g\",\n"
      << "  \"wifi_rate\": \"ErpOfdmRate6Mbps\",\n"
      << "  \"routing\": \"" << g_routingStrategy << "\",\n"
      << "  \"route_plan\": \"" << g_routePlanPath << "\",\n"
      << "  \"prediction_target\": \"" << g_predictionTarget << "\",\n"
      << "  \"prediction_horizon\": " << g_predictionHorizon << ",\n"
      << "  \"cost_mode\": \"" << g_costMode << "\",\n"
      << "  \"data_flow_enabled\": " << (g_enableDataFlow ? "true" : "false") << ",\n"
      << "  \"app_rate_kbps\": " << g_appRateKbps << ",\n"
      << "  \"app_packet_size\": " << g_appPacketSize << ",\n"
      << "  \"output_dir\": \"" << g_outputDir << "\"\n"
      << "}\n";
}

int main(int argc, char *argv[])
{
    std::string runName = "ns3_run";

    CommandLine cmd(__FILE__);
    cmd.AddValue("runName", "Run name (for scenario.json)", runName);
    cmd.AddValue("numUavs", "Number of UAV nodes", g_numUavs);
    cmd.AddValue("timeSteps", "Number of 1s snapshots", g_timeSteps);
    cmd.AddValue("seed", "RNG seed", g_seed);
    cmd.AddValue("mobility", "gauss-markov | random-waypoint", g_mobility);
    cmd.AddValue("xMax", "Area x size (m)", g_xMax);
    cmd.AddValue("yMax", "Area y size (m)", g_yMax);
    cmd.AddValue("zMin", "Min altitude (m)", g_zMin);
    cmd.AddValue("zMax", "Max altitude (m)", g_zMax);
    cmd.AddValue("commRange", "Target communication range (m) -> RxSensitivity", g_commRange);
    cmd.AddValue("txPower", "Tx power (dBm)", g_txPowerDbm);
    cmd.AddValue("refLoss", "Reference path loss at 1m (dB)", g_refLossDb);
    cmd.AddValue("pathLossExp", "Path loss exponent", g_pathLossExp);
    cmd.AddValue("noiseFloor", "Noise floor (dBm) for SNR feature", g_noiseFloorDbm);
    cmd.AddValue("gmAlpha", "Gauss-Markov memory alpha", g_gmAlpha);
    cmd.AddValue("rwpSpeedMin", "RWP speed min (m/s)", g_rwpSpeedMin);
    cmd.AddValue("rwpSpeedMax", "RWP speed max (m/s)", g_rwpSpeedMax);
    cmd.AddValue("warmup", "Warm-up before first snapshot (s)", g_warmup);
    cmd.AddValue("sourceId", "Traffic log source node", g_sourceId);
    cmd.AddValue("destId", "Traffic log destination node", g_destId);
    cmd.AddValue("outputDir", "Directory for CSV output", g_outputDir);
    cmd.AddValue("enableAnim", "Generate NetAnim XML (true/false)", g_enableAnim);
    cmd.AddValue("enableDataFlow", "Send a measured unicast UDP flow", g_enableDataFlow);
    cmd.AddValue("routePlan", "CSV route plan; also defines the evaluation window", g_routePlanPath);
    cmd.AddValue("routingStrategy", "olsr | hop | delay | persistence | logreg | xgb | edge-sage",
                 g_routingStrategy);
    cmd.AddValue("predictionTarget", "qos | survival | none", g_predictionTarget);
    cmd.AddValue("predictionHorizon", "Prediction horizon k", g_predictionHorizon);
    cmd.AddValue("costMode", "neglog | one-minus | none", g_costMode);
    cmd.AddValue("appRateKbps", "Offered UDP data rate (kbps)", g_appRateKbps);
    cmd.AddValue("appPacketSize", "UDP payload size (bytes)", g_appPacketSize);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(g_numUavs < 2, "numUavs must be at least 2");
    if (g_destId >= g_numUavs)
    {
        g_destId = g_numUavs - 1;
    }
    NS_ABORT_MSG_IF(g_sourceId >= g_numUavs || g_sourceId == g_destId,
                    "sourceId and destId must identify different UAVs");
    bool supportedStrategy = g_routingStrategy == "olsr" || g_routingStrategy == "hop" ||
                             g_routingStrategy == "delay" || g_routingStrategy == "persistence" ||
                             g_routingStrategy == "logreg" || g_routingStrategy == "xgb" ||
                             g_routingStrategy == "edge-sage";
    NS_ABORT_MSG_IF(!supportedStrategy, "Unsupported routing strategy: " << g_routingStrategy);
    NS_ABORT_MSG_IF(g_enableDataFlow && g_routePlanPath.empty(),
                    "enableDataFlow requires routePlan to define the evaluation window");
    NS_ABORT_MSG_IF(g_enableDataFlow && (g_appRateKbps <= 0.0 || g_appPacketSize == 0),
                    "appRateKbps and appPacketSize must be positive");
    if (!g_routePlanPath.empty())
    {
        LoadRoutePlan(g_routePlanPath);
    }

    RngSeedManager::SetSeed(g_seed == 0 ? 1 : g_seed);
    RngSeedManager::SetRun(1);

    g_nodes.Create(g_numUavs);
    g_prevDegree.assign(g_numUavs, 0);

    MobilityHelper mobility;
    std::ostringstream xRv;
    xRv << "ns3::UniformRandomVariable[Min=0.0|Max=" << g_xMax << "]";
    std::ostringstream yRv;
    yRv << "ns3::UniformRandomVariable[Min=0.0|Max=" << g_yMax << "]";
    std::ostringstream zRv;
    zRv << "ns3::UniformRandomVariable[Min=" << g_zMin << "|Max=" << g_zMax << "]";

    mobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                  "X", StringValue(xRv.str()),
                                  "Y", StringValue(yRv.str()),
                                  "Z", StringValue(zRv.str()));

    if (g_mobility == "gauss-markov")
    {
        std::ostringstream meanVel;
        meanVel << "ns3::UniformRandomVariable[Min=" << g_rwpSpeedMin << "|Max=" << g_rwpSpeedMax
                << "]";
        mobility.SetMobilityModel(
            "ns3::GaussMarkovMobilityModel",
            "Bounds", BoxValue(Box(0.0, g_xMax, 0.0, g_yMax, g_zMin, g_zMax)),
            "TimeStep", TimeValue(Seconds(1.0)),
            "Alpha", DoubleValue(g_gmAlpha),
            "MeanVelocity", StringValue(meanVel.str()),
            "MeanDirection", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.2831853]"),
            "MeanPitch", StringValue("ns3::UniformRandomVariable[Min=-0.3|Max=0.3]"),
            "NormalVelocity",
            StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=2.0|Bound=4.0]"),
            "NormalDirection",
            StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=0.4|Bound=0.8]"),
            "NormalPitch",
            StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=0.06|Bound=0.12]"));
        mobility.Install(g_nodes);
    }
    else
    {
        std::ostringstream speedRv;
        speedRv << "ns3::UniformRandomVariable[Min=" << g_rwpSpeedMin << "|Max=" << g_rwpSpeedMax
                << "]";
        ObjectFactory posFactory;
        posFactory.SetTypeId("ns3::RandomBoxPositionAllocator");
        posFactory.Set("X", StringValue(xRv.str()));
        posFactory.Set("Y", StringValue(yRv.str()));
        posFactory.Set("Z", StringValue(zRv.str()));
        Ptr<PositionAllocator> wpAlloc = posFactory.Create()->GetObject<PositionAllocator>();

        mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed", StringValue(speedRv.str()),
                                  "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
                                  "PositionAllocator", PointerValue(wpAlloc));
        mobility.Install(g_nodes);
    }

    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent", DoubleValue(g_pathLossExp),
                               "ReferenceDistance", DoubleValue(1.0),
                               "ReferenceLoss", DoubleValue(g_refLossDb));
    // Nakagami fast fading so edge-of-range links degrade gradually instead of
    // dropping at the sensitivity cutoff. m decreases with distance; the ns-3
    // defaults (m1=m2=0.75) are harsher than Rayleigh and overload tau_loss.
    channel.AddPropagationLoss("ns3::NakagamiPropagationLossModel",
                               "Distance1", DoubleValue(g_commRange * 0.4),
                               "Distance2", DoubleValue(g_commRange * 0.75),
                               "m0", DoubleValue(3.0),
                               "m1", DoubleValue(1.5),
                               "m2", DoubleValue(1.0));
    double rxSens = g_txPowerDbm - (g_refLossDb + 10.0 * g_pathLossExp * std::log10(g_commRange));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(g_txPowerDbm));
    phy.Set("TxPowerEnd", DoubleValue(g_txPowerDbm));
    phy.Set("RxSensitivity", DoubleValue(rxSens));
    phy.Set("CcaEdThreshold", DoubleValue(rxSens + 3.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("ErpOfdmRate6Mbps"),
                                 "ControlMode", StringValue("ErpOfdmRate6Mbps"));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(phy, mac, g_nodes);

    for (uint32_t i = 0; i < devices.GetN(); ++i)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(devices.Get(i));
        g_macToNode[Mac48Address::ConvertFrom(dev->GetAddress())] = i;
    }

    OlsrHelper olsrHelper;
    Ipv4StaticRoutingHelper staticRouting;
    Ipv4ListRoutingHelper listRouting;
    listRouting.Add(staticRouting, 20);
    listRouting.Add(olsrHelper, 10);

    InternetStackHelper stack;
    stack.SetRoutingHelper(listRouting);
    stack.Install(g_nodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.0.0.0", "255.255.255.0");
    g_ifaces = addr.Assign(devices);
    for (uint32_t i = 0; i < g_numUavs; ++i)
    {
        g_ipToNode[g_ifaces.GetAddress(i)] = i;
    }

    TypeId udpTid = TypeId::LookupByName("ns3::UdpSocketFactory");
    g_txSockets.resize(g_numUavs);
    for (uint32_t i = 0; i < g_numUavs; ++i)
    {
        Ptr<Socket> rx = Socket::CreateSocket(g_nodes.Get(i), udpTid);
        rx->Bind(InetSocketAddress(Ipv4Address::GetAny(), PROBE_PORT));
        rx->SetRecvCallback(MakeCallback(&ReceiveProbe));

        Ptr<Socket> tx = Socket::CreateSocket(g_nodes.Get(i), udpTid);
        tx->SetAllowBroadcast(true);
        tx->Connect(InetSocketAddress(Ipv4Address("255.255.255.255"), PROBE_PORT));
        g_txSockets[i] = tx;

        double start = 0.1 + i * (PROBE_INTERVAL / g_numUavs);
        Simulator::Schedule(Seconds(start), &SendProbe, i);
    }

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
                    MakeCallback(&MonitorSniffRx));

    FlowMonitorHelper flowMonitorHelper;
    Ptr<FlowMonitor> flowMonitor;
    if (g_enableDataFlow)
    {
        double firstPlanTime = g_routePlan.begin()->first;
        double lastPlanTime = g_routePlan.rbegin()->first;
        double appStart = g_warmup + firstPlanTime + 1.01;
        double appStop = g_warmup + lastPlanTime + 2.0;
        double intervalSeconds = g_appPacketSize * 8.0 / (g_appRateKbps * 1000.0);

        UdpServerHelper server(DATA_PORT);
        ApplicationContainer serverApp = server.Install(g_nodes.Get(g_destId));
        serverApp.Start(Seconds(appStart - 0.1));
        serverApp.Stop(Seconds(appStop + 0.1));

        UdpClientHelper client(g_ifaces.GetAddress(g_destId), DATA_PORT);
        client.SetAttribute("MaxPackets", UintegerValue(std::numeric_limits<uint32_t>::max()));
        client.SetAttribute("Interval", TimeValue(Seconds(intervalSeconds)));
        client.SetAttribute("PacketSize", UintegerValue(g_appPacketSize));
        ApplicationContainer clientApp = client.Install(g_nodes.Get(g_sourceId));
        clientApp.Start(Seconds(appStart));
        clientApp.Stop(Seconds(appStop));
        flowMonitor = flowMonitorHelper.InstallAll();

        if (g_routingStrategy != "olsr")
        {
            for (const auto &[step, entry] : g_routePlan)
            {
                (void)entry;
                Simulator::Schedule(Seconds(g_warmup + step + 1.001), &ApplyRoutePlan, step);
            }
        }
    }

    g_nodesCsv.open(g_outputDir + "/nodes.csv");
    g_edgesCsv.open(g_outputDir + "/edges.csv");
    g_trafficCsv.open(g_outputDir + "/traffic_log.csv");
    g_nodesCsv << std::fixed << std::setprecision(4)
               << "time,node_id,x,y,z,vx,vy,vz,speed,degree\n";
    g_edgesCsv << std::fixed << std::setprecision(4)
               << "time,src,dst,distance,connected,relative_speed,rssi,snr,delay,packet_loss,"
                  "throughput,p_stable,weight\n";
    g_trafficCsv << std::fixed << std::setprecision(4)
                 << "time,source,destination,reachable,route_path,num_edges,routing_protocol,"
                    "olsr_mpr_nodes,olsr_avg_rt_size\n";

    for (uint32_t t = 0; t < g_timeSteps; ++t)
    {
        Simulator::Schedule(Seconds(g_warmup + t + 1.0), &TakeSnapshot, t);
    }
    Simulator::Schedule(Seconds(g_warmup), []
                        { g_window.clear(); });

    AnimationInterface *anim = nullptr;
    if (g_enableAnim)
    {
        std::string animFile = g_outputDir + "/uav-animation.xml";
        anim = new AnimationInterface(animFile);
        anim->SetMobilityPollInterval(Seconds(0.25));
        anim->EnablePacketMetadata(true);
        for (uint32_t i = 0; i < g_numUavs; ++i)
        {
            std::ostringstream desc;
            desc << "UAV-" << i;
            anim->UpdateNodeDescription(i, desc.str());
            if (i == g_sourceId)
            {
                anim->UpdateNodeColor(i, 22, 163, 74);
                anim->UpdateNodeSize(i, 8.0, 8.0);
            }
            else if (i == g_destId)
            {
                anim->UpdateNodeColor(i, 220, 38, 38);
                anim->UpdateNodeSize(i, 8.0, 8.0);
            }
            else
            {
                anim->UpdateNodeColor(i, 14, 165, 233);
                anim->UpdateNodeSize(i, 6.0, 6.0);
            }
        }
        std::cout << "[ANIM] NetAnim XML -> " << animFile << std::endl;
    }

    Simulator::Stop(Seconds(g_warmup + g_timeSteps + 1.0));
    Simulator::Run();
    if (g_enableDataFlow)
    {
        WriteClosedLoopMetrics(flowMonitor, flowMonitorHelper);
    }
    Simulator::Destroy();

    delete anim;

    g_nodesCsv.close();
    g_edgesCsv.close();
    g_trafficCsv.close();
    WriteScenarioJson(runName);

    std::cout << "[OK] ns-3 dataset written to " << g_outputDir << std::endl;
    return 0;
}
