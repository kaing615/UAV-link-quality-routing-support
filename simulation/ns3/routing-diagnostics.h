// Opt-in, read-only ns-3.45 observers. Included after GetOlsr: no packet tags,
// random draws, traffic scheduling or protocol changes are introduced here.
#include <set>
#include <string>
#include <vector>
#include <tuple>
#include <cstdint>

struct DiagnosticRouteStep { bool protocol=false, entry=false; int64_t next=-1; };
struct DiagnosticRouteResult
{
    std::vector<uint32_t> path;
    std::string reason="hop_limit";
    int64_t failing=-1;
};

// A bounded, independently testable read-only traversal. The lookup reports
// only measured protocol state, including undecodable next-hop addresses.
template<class Lookup>
static DiagnosticRouteResult TraverseDiagnosticRoute(uint32_t source,uint32_t destination,
                                                      uint32_t hopLimit,Lookup lookup)
{
    DiagnosticRouteResult result;
    uint32_t current=source;
    std::set<uint32_t> visited;
    result.path.push_back(current);
    for (uint32_t hop=0;hop<=hopLimit;++hop)
    {
        result.failing=current;
        if (current==destination) { result.reason="success"; result.failing=-1; break; }
        if (!visited.insert(current).second) { result.reason="loop"; break; }
        if (hop==hopLimit) break;
        const auto step=lookup(current);
        if (!step.protocol) { result.reason="missing_protocol"; break; }
        if (!step.entry) { result.reason="missing_entry"; break; }
        if (step.next<0) { result.reason="unknown_next"; break; }
        current=static_cast<uint32_t>(step.next);
        result.path.push_back(current);
    }
    return result;
}

#ifndef UAV_DIAGNOSTICS_TRAVERSAL_ONLY
static bool g_enableRoutingDiagnostics = false;
static double g_sanitySpacingFraction = 0.55;
static std::ofstream g_routingEntries, g_routeTraversal, g_olsrState, g_dropCounters;
static std::ofstream g_directedProbes;
static std::map<uint32_t,uint64_t> g_diagnosticProbeTx;
using DropKey = std::tuple<int64_t, uint32_t, std::string, std::string, std::string>;
static std::map<DropKey, uint64_t> g_diagnosticDrops;

static void GuardDiagnosticOutput()
{
    if (!g_enableRoutingDiagnostics && g_scenarioMode != "routing-sanity") return;
    // The existing prevention bridge already validates and consumes its fresh
    // reservation. No other diagnostic invocation may reuse existing files.
    if (g_scenarioMode == "relay-prevention") return;
    const std::filesystem::path dir(g_outputDir);
    std::error_code error;
    const bool exists = std::filesystem::exists(dir, error);
    NS_ABORT_MSG_IF(error, "Cannot inspect diagnostic output: " << error.message());
    if (!exists) std::filesystem::create_directories(dir, error);
    NS_ABORT_MSG_IF(error || !std::filesystem::is_directory(dir) || !std::filesystem::is_empty(dir),
                    "Diagnostic output directory must be empty/new");
}

static void ConfigureSanityMobility()
{
    const double spacing = g_sanitySpacingFraction * g_commRange;
    NS_ABORT_MSG_IF((g_numUavs-1)*spacing > g_xMax, "routing-sanity chain exceeds xMax");
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_nodes);
    for (uint32_t i=0; i<g_numUavs; ++i)
        g_nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(
            g_xMax/2 + (static_cast<double>(i)-(g_numUavs-1)/2.0)*spacing,
            g_yMax/2, (g_zMin+g_zMax)/2));
}

// Decode only headers whose framing is known from each trace's installed API.
// All removes occur on a copy; arbitrary undecodable bytes stay unknown.
static std::string DiagnosticUdp(Ptr<const Packet> packet, const Ipv4Header& ip)
{
    if (ip.GetProtocol()!=17 || ip.GetFragmentOffset()!=0 || packet->GetSize()<8) return "unknown";
    UdpHeader udp;
    if (packet->PeekHeader(udp)!=8) return "unknown";
    const uint16_t port=udp.GetDestinationPort();
    if (port==DATA_PORT || (port>=BACKGROUND_PORT_BASE &&
        static_cast<uint32_t>(port-BACKGROUND_PORT_BASE)<g_backgroundFlows)) return "data";
    if (port==PROBE_PORT) return "probe";
    if (port==698) return "OLSR";
    return "unknown";
}

static std::string DiagnosticIp(Ptr<const Packet> packet)
{
    if (packet->GetSize()<20) return "unknown";
    auto copy=packet->Copy();
    uint8_t first=0; copy->CopyData(&first,1);
    if ((first>>4)!=4 || (first&15)<5 || copy->GetSize()<(first&15)*4u) return "unknown";
    Ipv4Header ip;
    if (!copy->RemoveHeader(ip)) return "unknown";
    return DiagnosticUdp(copy,ip);
}

static std::string DiagnosticLlc(Ptr<const Packet> packet)
{
    if (packet->GetSize()<8) return "unknown";
    uint8_t bytes[3]; packet->CopyData(bytes,3);
    if (bytes[0]!=0xaa || bytes[1]!=0xaa || bytes[2]!=3) return "unknown";
    auto copy=packet->Copy(); LlcSnapHeader llc; copy->RemoveHeader(llc);
    if (llc.GetType()==0x0806) return "ARP";
    return llc.GetType()==0x0800 ? DiagnosticIp(copy) : "unknown";
}

static void DiagnosticDrop(uint32_t node, const std::string& layer,
                           const std::string& reason, const std::string& traffic)
{
    const double elapsed=Simulator::Now().GetSeconds()-g_warmup;
    const int64_t window=elapsed<0 ? -1 : static_cast<int64_t>(std::floor(elapsed));
    ++g_diagnosticDrops[{window,node,layer,reason,traffic}];
}
static void DiagnosticIpDrop(uint32_t node, const Ipv4Header& ip, Ptr<const Packet> packet,
                             Ipv4L3Protocol::DropReason reason, Ptr<Ipv4>, uint32_t)
{ DiagnosticDrop(node,"IP",std::to_string(static_cast<unsigned>(reason)),DiagnosticUdp(packet,ip)); }
static void DiagnosticArpDrop(uint32_t node, Ptr<const Packet> packet)
{ DiagnosticDrop(node,"ARP","unspecified",DiagnosticIp(packet)); }
static void DiagnosticArpCacheDrop(uint32_t node, Ptr<const Packet> packet)
{ DiagnosticDrop(node,"ARP_cache","retry_exhausted",DiagnosticIp(packet)); }
static void DiagnosticMacDrop(uint32_t node, Ptr<const Packet> packet)
{ DiagnosticDrop(node,"MAC_tx","unspecified",DiagnosticLlc(packet)); }
static void DiagnosticMacRxDrop(uint32_t node, Ptr<const Packet> packet)
{ DiagnosticDrop(node,"MAC_rx","unspecified",DiagnosticLlc(packet)); }
static void DiagnosticMpduDrop(uint32_t node, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{ DiagnosticDrop(node,"MAC_mpdu",std::to_string(static_cast<unsigned>(reason)),
                 mpdu->GetHeader().IsData() ? DiagnosticLlc(mpdu->GetPacket()) : "unknown"); }
static std::string DiagnosticPhyTraffic(Ptr<const Packet> packet)
{
    if (packet->GetSize()<24) return "unknown";
    auto copy=packet->Copy(); WifiMacHeader wifi;
    if (!copy->RemoveHeader(wifi) || !wifi.IsData()) return "unknown";
    return DiagnosticLlc(copy);
}
static void DiagnosticPhyDrop(uint32_t node, Ptr<const Packet> packet, WifiPhyRxfailureReason reason)
{ DiagnosticDrop(node,"PHY_rx_observation",std::to_string(static_cast<unsigned>(reason)),DiagnosticPhyTraffic(packet)); }
static void DiagnosticPhyTxDrop(uint32_t node, Ptr<const Packet> packet)
{ DiagnosticDrop(node,"PHY_tx","unspecified",DiagnosticPhyTraffic(packet)); }
static void DiagnosticPhyTxBegin(uint32_t node, Ptr<const Packet> packet, double)
{ if (DiagnosticPhyTraffic(packet)=="probe") ++g_diagnosticProbeTx[node]; }

static void ConnectRoutingDiagnostics(const NetDeviceContainer& devices)
{
    if (!g_enableRoutingDiagnostics) return;
    for (uint32_t i=0;i<g_numUavs;++i)
    {
        auto ipv4=g_nodes.Get(i)->GetObject<Ipv4L3Protocol>();
        auto arp=g_nodes.Get(i)->GetObject<ArpL3Protocol>();
        auto device=DynamicCast<WifiNetDevice>(devices.Get(i));
        for (uint32_t interface=0;interface<ipv4->GetNInterfaces();++interface)
        {
            auto cache=ipv4->GetInterface(interface)->GetArpCache();
            if (cache) NS_ABORT_MSG_IF(!cache->TraceConnectWithoutContext("Drop",MakeBoundCallback(&DiagnosticArpCacheDrop,i)),
                                     "Cannot attach installed ARP cache drop trace");
        }
        NS_ABORT_MSG_IF(!ipv4->TraceConnectWithoutContext("Drop",MakeBoundCallback(&DiagnosticIpDrop,i)) ||
            !arp->TraceConnectWithoutContext("Drop",MakeBoundCallback(&DiagnosticArpDrop,i)) ||
            !device->GetMac()->TraceConnectWithoutContext("MacTxDrop",MakeBoundCallback(&DiagnosticMacDrop,i)) ||
            !device->GetMac()->TraceConnectWithoutContext("MacRxDrop",MakeBoundCallback(&DiagnosticMacRxDrop,i)) ||
            !device->GetMac()->TraceConnectWithoutContext("DroppedMpdu",MakeBoundCallback(&DiagnosticMpduDrop,i)) ||
            !device->GetPhy()->TraceConnectWithoutContext("PhyRxDrop",MakeBoundCallback(&DiagnosticPhyDrop,i)) ||
            !device->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&DiagnosticPhyTxBegin,i)) ||
            !device->GetPhy()->TraceConnectWithoutContext("PhyTxDrop",MakeBoundCallback(&DiagnosticPhyTxDrop,i)),
            "Cannot attach installed ns-3.45 diagnostic traces");
    }
}

static void OpenRoutingDiagnostics()
{
    if (!g_enableRoutingDiagnostics) return;
    g_routingEntries.open(g_outputDir+"/routing_entries_v1.csv");
    g_routeTraversal.open(g_outputDir+"/route_traversal_v1.csv");
    g_olsrState.open(g_outputDir+"/olsr_state_v1.csv");
    g_dropCounters.open(g_outputDir+"/drop_counters_v1.csv");
    g_directedProbes.open(g_outputDir+"/directed_probes_v1.csv");
    NS_ABORT_MSG_IF(!g_routingEntries || !g_routeTraversal || !g_olsrState || !g_dropCounters || !g_directedProbes,"Cannot open diagnostic outputs");
    g_routingEntries << "schema_version,time,sim_time_s,node_id,protocol_present,destination_ip,entry_exists,next_ip,next_node,interface,hops,neighbor_count,twohop_count,topology_count,mpr_count,mpr_selector_count\n";
    g_routeTraversal << "schema_version,time,sim_time_s,source,destination,success,reason,failing_node,partial_path\n";
    g_olsrState << "schema_version,time,sim_time_s,node_id,kind,address1,address2,status,willingness,sequence,expires_s\n";
    g_dropCounters << "schema_version,time,window_start_s,window_end_s,node_id,layer,reason,traffic_class,count\n";
    g_directedProbes << "schema_version,time,window_start_s,window_end_s,source,destination,phy_tx_probe_count,rx_probe_count\n";
}

static void SnapshotRoutingDiagnostics(uint32_t step)
{
    if (!g_enableRoutingDiagnostics) return;
    auto destination=g_ifaces.GetAddress(g_destId);
    for (uint32_t i=0;i<g_numUavs;++i) for (uint32_t j=0;j<g_numUavs;++j) if (i!=j)
    {
        const auto rx=g_window.find({i,j});
        g_directedProbes << "1," << step << ',' << g_warmup+step << ',' << g_warmup+step+1 << ',' << i << ',' << j << ','
            << g_diagnosticProbeTx[i] << ',' << (rx==g_window.end()?0:rx->second.rxCount) << '\n';
    }
    g_diagnosticProbeTx.clear();
    for (uint32_t i=0;i<g_numUavs;++i)
    {
        auto o=GetOlsr(g_nodes.Get(i));
        g_routingEntries << "1," << step << ',' << Simulator::Now().GetSeconds() << ',' << i << ',' << bool(o) << ',' << destination << ',';
        bool found=false;
        if (o) for (const auto& e:o->GetRoutingTableEntries()) if (e.destAddr==destination)
        {
            auto next=g_ipToNode.find(e.nextAddr);
            g_routingEntries << "1," << e.nextAddr << ',';
            if (next!=g_ipToNode.end()) g_routingEntries << next->second;
            g_routingEntries << ',' << e.interface << ',' << e.distance;
            found=true; break;
        }
        if (!found) g_routingEntries << "0,,,,";
        g_routingEntries << ',' << (o?o->GetNeighbors().size():0) << ',' << (o?o->GetTwoHopNeighbors().size():0)
            << ',' << (o?o->GetTopologySet().size():0) << ',' << (o?o->GetMprSet().size():0)
            << ',' << (o?o->GetMprSelectors().size():0) << '\n';
        if (!o) continue;
        auto prefix=[&](const char* kind) { g_olsrState << "1," << step << ',' << Simulator::Now().GetSeconds() << ',' << i << ',' << kind << ','; };
        for (const auto& e:o->GetNeighbors()) { prefix("neighbor"); g_olsrState << e.neighborMainAddr << ",," << unsigned(e.status) << ',' << unsigned(e.willingness) << ",,\n"; }
        for (const auto& e:o->GetTwoHopNeighbors()) { prefix("twohop"); g_olsrState << e.neighborMainAddr << ',' << e.twoHopNeighborAddr << ",,,," << e.expirationTime.GetSeconds() << '\n'; }
        for (const auto& e:o->GetTopologySet()) { prefix("topology"); g_olsrState << e.destAddr << ',' << e.lastAddr << ",,," << e.sequenceNumber << ',' << e.expirationTime.GetSeconds() << '\n'; }
        for (const auto& e:o->GetMprSet()) { prefix("mpr"); g_olsrState << e << ",,,,,\n"; }
        for (const auto& e:o->GetMprSelectors()) { prefix("mpr_selector"); g_olsrState << e.mainAddr << ",,,,," << e.expirationTime.GetSeconds() << '\n'; }
    }
    const auto traversal=TraverseDiagnosticRoute(g_sourceId,g_destId,g_numUavs,[&](uint32_t current)
    {
        DiagnosticRouteStep result;
        auto o=GetOlsr(g_nodes.Get(current));
        if (!o) return result;
        result.protocol=true;
        for (const auto& e:o->GetRoutingTableEntries()) if (e.destAddr==destination)
        {
            result.entry=true; auto next=g_ipToNode.find(e.nextAddr);
            if (next!=g_ipToNode.end()) result.next=next->second;
            break;
        }
        return result;
    });
    g_routeTraversal << "1," << step << ',' << Simulator::Now().GetSeconds() << ',' << g_sourceId << ',' << g_destId << ',' << (traversal.reason=="success") << ',' << traversal.reason << ',';
    if (traversal.failing>=0) g_routeTraversal << traversal.failing;
    g_routeTraversal << ',';
    for (size_t i=0;i<traversal.path.size();++i) g_routeTraversal << (i?"->":"") << traversal.path[i];
    g_routeTraversal << '\n';
}
static void WriteDiagnosticDrops()
{
    if (!g_enableRoutingDiagnostics) return;
    for (const auto& [key,count]:g_diagnosticDrops)
    {
        const auto& [window,node,layer,reason,traffic]=key;
        const double start=window<0?0:g_warmup+window, end=window<0?g_warmup:std::min(g_warmup+window+1,Simulator::Now().GetSeconds());
        g_dropCounters << "1," << window << ',' << start << ',' << end << ',' << node << ',' << layer << ',' << reason << ',' << traffic << ',' << count << '\n';
    }
}
#endif
