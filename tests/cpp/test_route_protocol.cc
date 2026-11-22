#include "route-protocol.h"
#include <cassert>
#include <sstream>

int main()
{
    // 4 nodes: source 0, destination 3, relays 1 and 2 -> 6 velocity values.
    std::istringstream input("UAVACT 2 5 0 0 0 0.5 0 0 ROUTE 3 0 2 3\nUAVACT 2 6 0 0 0 0 0 0 ROUTE 0\nUAVQUIT\n");
    auto a = recovery::ReadRoutedAction(input, 5, 6, 4, 0, 3);
    assert(!a.quit && a.values.size()==6 && a.values[3]==0.5 && (a.route==std::vector<uint32_t>{0,2,3}));
    auto b = recovery::ReadRoutedAction(input, 6, 6, 4, 0, 3);
    assert(!b.quit && b.route.empty());
    assert(recovery::ReadRoutedAction(input, 7, 6, 4, 0, 3).quit);
    for (const std::string line : {
             "UAVACT 1 5 0 0 0 0 0 0 ROUTE 0",          // v1 line on the v2 channel
             "UAVACT 2 4 0 0 0 0 0 0 ROUTE 0",          // wrong step
             "UAVACT 2 5 0 0 0 0 0 0",                  // missing ROUTE
             "UAVACT 2 5 0 0 0 0 0 0 ROUTE 1 0",        // single node
             "UAVACT 2 5 0 0 0 0 0 0 ROUTE 2 1 3",      // does not start at source
             "UAVACT 2 5 0 0 0 0 0 0 ROUTE 3 0 3 3",    // repeated node
             "UAVACT 2 5 0 0 0 0 0 0 ROUTE 3 0 9 3",    // unknown node
             "UAVACT 2 5 0 0 0 0 0 0 ROUTE 2 0 3 7",    // extra token
             "UAVACT 2 5 2 0 0 0 0 0 ROUTE 0"})         // velocity out of range
    {
        bool failed=false;
        try { std::istringstream bad(line); recovery::ReadRoutedAction(bad,5,6,4,0,3); }
        catch (const std::runtime_error &) { failed=true; }
        assert(failed);
    }
    // v1 parser is unchanged and still rejects v2 lines.
    std::istringstream v2("UAVACT 2 5 0 0 0 0 0 0 ROUTE 0\n");
    bool failed=false;
    try { recovery::ReadAction(v2,5,6); } catch (const std::runtime_error &) { failed=true; }
    assert(failed);
}
