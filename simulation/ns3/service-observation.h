// Opt-in causal service counters; no events, random draws, packet mutation or
// routing writes. FlowMonitor read snapshots never call CheckForLostPackets.
#include <cstdint>
static bool ServiceFlowAddressMatch(uint32_t from,uint32_t to,uint16_t fromPort,uint16_t toPort,
                                   uint32_t source,uint32_t destination,uint16_t sourcePort,
                                   bool sourceWildcard,bool destinationWildcard)
{
    return (from==source || (sourceWildcard && from==0)) &&
           (to==destination || (destinationWildcard && to==0)) && toPort==9000 &&
           fromPort>0 && (sourcePort==0 || fromPort==sourcePort);
}
#ifndef UAV_SERVICE_FILTER_ONLY
static Ptr<FlowMonitor> g_serviceMonitor;
static Ptr<Ipv4FlowClassifier> g_serviceClassifier;
static uint64_t g_serviceOffered=0, g_serviceReceived=0;
static uint16_t g_serviceSourcePort=0;
static uint64_t g_servicePrevious[3]={0,0,0};
static double g_servicePreviousTime=0;

static bool ServiceAddresses(const Address& from,const Address& to,bool offered)
{
    if (!InetSocketAddress::IsMatchingType(from) || !InetSocketAddress::IsMatchingType(to)) return false;
    const auto source=InetSocketAddress::ConvertFrom(from), destination=InetSocketAddress::ConvertFrom(to);
    // UdpClient/UdpServer bind wildcard sockets; callbacks are connected ONLY
    // to the configured source client/destination server objects. Remote IPs
    // and ports remain checked, and sink source-port must match that client.
    return ServiceFlowAddressMatch(source.GetIpv4().Get(),destination.GetIpv4().Get(),
        source.GetPort(),destination.GetPort(),g_ifaces.GetAddress(g_sourceId).Get(),
        g_ifaces.GetAddress(g_destId).Get(),offered?0:g_serviceSourcePort,offered,!offered);
}
static void ServiceOffered(Ptr<const Packet>,const Address& from,const Address& to)
{
    if (!ServiceAddresses(from,to,true)) return;
    g_serviceSourcePort=InetSocketAddress::ConvertFrom(from).GetPort();
    ++g_serviceOffered;
}
static void ServiceReceived(Ptr<const Packet>,const Address& from,const Address& to)
{
    if (ServiceAddresses(from,to,false) && g_serviceSourcePort>0)
        ++g_serviceReceived;
}
static std::pair<uint64_t,uint64_t> ServiceIpCounts()
{
    uint64_t tx=0,rx=0;
    if (g_serviceMonitor && g_serviceClassifier) for (const auto& [id,stats]:g_serviceMonitor->GetFlowStats())
    {
        const auto flow=g_serviceClassifier->FindFlow(id);
        if (flow.protocol==17 && flow.sourceAddress==g_ifaces.GetAddress(g_sourceId) &&
            flow.destinationAddress==g_ifaces.GetAddress(g_destId) && flow.destinationPort==DATA_PORT &&
            flow.sourcePort==g_serviceSourcePort) { tx+=stats.txPackets; rx+=stats.rxPackets; }
    }
    return {tx,rx};
}
static void AppendServiceObservation(std::ostream& out)
{
    if (!g_enableServiceObservation) return;
    const auto ip=ServiceIpCounts();
    const uint64_t counts[3]={g_serviceOffered,ip.first,g_serviceReceived};
    const char* names[3]={"offered_packets","source_tx_packets","sink_rx_packets"};
    const double now=Simulator::Now().GetSeconds();
    out << ",\"service_observation\":{\"schema\":\"service-observation-v1\",\"sim_time_s\":" << now
        << ",\"window_start_s\":" << g_servicePreviousTime << ",\"window_end_s\":" << now
        << ",\"flow\":{\"source\":" << g_sourceId << ",\"destination\":" << g_destId
        << ",\"destination_port\":" << DATA_PORT << ",\"protocol\":\"udp\"}";
    for (unsigned group=0;group<2;++group)
    {
        out << (group?",\"window\":{":",\"cumulative\":{");
        for (unsigned i=0;i<3;++i) out << (i?",":"") << '"' << names[i] << "\":" << (group?counts[i]-g_servicePrevious[i]:counts[i]);
        out << '}';
    }
    const auto route=TraverseDiagnosticRoute(g_sourceId,g_destId,g_numUavs,[&](uint32_t node)
    {
        DiagnosticRouteStep result; auto olsr=GetOlsr(g_nodes.Get(node));
        if (!olsr) return result;
        result.protocol=true;
        for (const auto& e:olsr->GetRoutingTableEntries()) if (e.destAddr==g_ifaces.GetAddress(g_destId))
        { result.entry=true; auto next=g_ipToNode.find(e.nextAddr); if (next!=g_ipToNode.end()) result.next=next->second; break; }
        return result;
    });
    out << ",\"route\":{\"nodes\":[";
    for (size_t i=0;i<route.path.size();++i) out << (i?",":"") << route.path[i];
    out << "],\"success\":" << (route.reason=="success"?"true":"false") << ",\"reason\":\"" << route.reason << "\",\"failing_node\":";
    if (route.failing<0) out << "null"; else out << route.failing;
    out << "},\"route_probe_hops\":[";
    for (size_t i=1;i<route.path.size();++i)
    {
        const auto a=route.path[i-1],b=route.path[i];
        // g_window[{transmitter,receiver}].rxCount is populated by ProbeReceive.
        // find() avoids adding observations to the shared map.
        const auto f=g_window.find({a,b}),r=g_window.find({b,a});
        const double forward=f==g_window.end()?0:Clamp01(f->second.rxCount*PROBE_INTERVAL);
        const double reverse=r==g_window.end()?0:Clamp01(r->second.rxCount*PROBE_INTERVAL);
        out << (i>1?",":"") << "{\"from\":" << a << ",\"to\":" << b
            << ",\"rx_ratio_forward\":" << forward << ",\"rx_ratio_reverse\":" << reverse << '}';
    }
    out << "]}";
    for (unsigned i=0;i<3;++i) g_servicePrevious[i]=counts[i];
    g_servicePreviousTime=now;
}
static void WriteFinalServiceAccounting()
{
    if (!g_enableServiceObservation) return;
    const auto ip=ServiceIpCounts();
    std::ofstream out(g_outputDir+"/service_final_accounting.json");
    NS_ABORT_MSG_IF(!out,"Cannot open final service accounting");
    out << std::setprecision(12) << "{\"schema\":\"service-final-accounting-v1\",\"counting_origin_s\":0,\"sim_time_s\":" << Simulator::Now().GetSeconds()
        << ",\"last_live_frame_time_s\":" << g_servicePreviousTime
        << ",\"offered_packets\":" << g_serviceOffered << ",\"source_tx_packets\":" << ip.first
        << ",\"sink_rx_packets\":" << g_serviceReceived << ",\"flowmonitor_rx_packets\":" << ip.second
        << ",\"late_sink_rx_after_last_frame\":" << g_serviceReceived-g_servicePrevious[2]
        << ",\"offered_after_last_frame\":" << g_serviceOffered-g_servicePrevious[0]
        << ",\"source_tx_after_last_frame\":" << ip.first-g_servicePrevious[1]
        << ",\"ordering\":\"last live snapshot before controller UAVQUIT; final record after Simulator::Run and existing final FlowMonitor metrics; no artificial flush\"}\n";
}
#endif // UAV_SERVICE_FILTER_ONLY
