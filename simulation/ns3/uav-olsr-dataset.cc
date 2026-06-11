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
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/wifi-module.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
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

static constexpr double PROBE_INTERVAL = 0.05;
static constexpr uint16_t PROBE_PORT = 9999;

// ---------------------------------------------------------------------------
struct LinkWindow
{
    uint32_t rxCount = 0;
    double delaySumMs = 0.0;
    uint32_t rssiCount = 0;
    double rssiSumDbm = 0.0;
};

static NodeContainer g_nodes;
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
    bool reachable = WalkRoute(g_sourceId, g_destId, path);
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
    for (uint32_t i = 0; i < n; ++i)
    {
        Ptr<olsr::RoutingProtocol> o = GetOlsr(g_nodes.Get(i));
        rtSizeSum += o ? o->GetRoutingTableEntries().size() : 0;
    }
    g_trafficCsv << step << ',' << g_sourceId << ',' << g_destId << ',' << (reachable ? 1 : 0)
                 << ',' << pathStr.str() << ',' << numConnectedEdges << ",olsr(ns3),0,"
                 << rtSizeSum / n << '\n';

    g_window.clear();
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
      << "  \"routing\": \"ns3::olsr\",\n"
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
    cmd.Parse(argc, argv);

    if (g_destId >= g_numUavs)
    {
        g_destId = g_numUavs - 1;
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
    Ipv4ListRoutingHelper listRouting;
    listRouting.Add(olsrHelper, 10);

    InternetStackHelper stack;
    stack.SetRoutingHelper(listRouting);
    stack.Install(g_nodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer ifaces = addr.Assign(devices);
    for (uint32_t i = 0; i < g_numUavs; ++i)
    {
        g_ipToNode[ifaces.GetAddress(i)] = i;
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

    Simulator::Stop(Seconds(g_warmup + g_timeSteps + 1.0));
    Simulator::Run();
    Simulator::Destroy();

    g_nodesCsv.close();
    g_edgesCsv.close();
    g_trafficCsv.close();
    WriteScenarioJson(runName);

    std::cout << "[OK] ns-3 dataset written to " << g_outputDir << std::endl;
    return 0;
}
