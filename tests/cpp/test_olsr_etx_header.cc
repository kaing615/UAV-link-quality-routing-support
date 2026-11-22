// Round trip of the OLSR-ETX wire format (simulation/ns3/patches/ns-3.45-olsr-etx.patch).
// Build against the patched ns-3 prefix, e.g.
//   g++ -std=c++20 -I$P/include tests/cpp/test_olsr_etx_header.cc -L$P/lib \
//       -lns3.45-olsr-default -lns3.45-internet-default -lns3.45-network-default -lns3.45-core-default
#include "ns3/olsr-header.h"
#include "ns3/packet.h"

#include <cassert>

using namespace ns3;
using olsr::MessageHeader;

static MessageHeader
Hello(bool lq)
{
    MessageHeader msg;
    msg.SetVTime(Seconds(6));
    msg.SetOriginatorAddress(Ipv4Address("10.0.0.1"));
    msg.SetTimeToLive(1);
    msg.SetMessageSequenceNumber(7);
    MessageHeader::Hello& hello = msg.GetHello();
    hello.SetHTime(Seconds(2));
    hello.willingness = olsr::Willingness::DEFAULT;
    hello.linkQuality = lq;
    MessageHeader::Hello::LinkMessage lm;
    lm.linkCode = 6;
    lm.neighborInterfaceAddresses = {Ipv4Address("10.0.0.2"), Ipv4Address("10.0.0.3")};
    if (lq)
    {
        lm.lq = {255, 128};
        lm.nlq = {200, 3};
    }
    hello.linkMessages.push_back(lm);
    return msg;
}

int
main()
{
    // Off: identical size to RFC 3626 (4 + 4 + 2 * 4 bytes body).
    assert(Hello(false).GetSerializedSize() == 12 + 4 + 4 + 2 * 4);
    // On: 4 extra bytes per advertised address.
    assert(Hello(true).GetSerializedSize() == 12 + 4 + 4 + 2 * 8);

    for (bool lq : {false, true})
    {
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(Hello(lq));
        uint8_t type = 0;
        packet->CopyData(&type, 1);
        assert(type == (lq ? 201 : 1)); // LQ_HELLO on the wire only when enabled
        MessageHeader read;
        packet->RemoveHeader(read);
        assert(read.GetMessageType() == MessageHeader::HELLO_MESSAGE); // in memory it stays HELLO
        const auto& lm = read.GetHello().linkMessages.at(0);
        assert(read.GetHello().linkQuality == lq);
        assert(lm.neighborInterfaceAddresses.size() == 2 && lm.neighborInterfaceAddresses[1] == Ipv4Address("10.0.0.3"));
        assert(lq ? (lm.lq == std::vector<uint8_t>{255, 128} && lm.nlq == std::vector<uint8_t>{200, 3}) : lm.lq.empty());
    }

    MessageHeader tc;
    tc.SetVTime(Seconds(15));
    tc.SetOriginatorAddress(Ipv4Address("10.0.0.4"));
    tc.SetTimeToLive(255);
    tc.SetMessageSequenceNumber(9);
    tc.GetTc().ansn = 3;
    tc.GetTc().linkQuality = true;
    tc.GetTc().neighborAddresses = {Ipv4Address("10.0.0.5")};
    tc.GetTc().lq = {77};
    tc.GetTc().nlq = {66};
    Ptr<Packet> packet = Create<Packet>();
    packet->AddHeader(tc);
    MessageHeader read;
    packet->RemoveHeader(read);
    assert(read.GetMessageType() == MessageHeader::TC_MESSAGE && read.GetTc().linkQuality);
    assert(read.GetTc().ansn == 3 && read.GetTc().lq.at(0) == 77 && read.GetTc().nlq.at(0) == 66);
    return 0;
}
