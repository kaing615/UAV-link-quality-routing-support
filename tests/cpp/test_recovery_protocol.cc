#include "recovery-protocol.h"
#include <cassert>
#include <sstream>

int main()
{
    std::istringstream input("UAVACT 1 3 0.5 -1 0\nUAVQUIT\n");
    auto action = recovery::ReadAction(input, 3, 3);
    assert(!action.quit && action.values.size()==3 && action.values[1]==-1);
    assert(recovery::ReadAction(input, 4, 3).quit);
    for (const std::string line : {"UAVACT 1 2 0 0 0", "UAVACT 2 3 0 0 0",
                                   "UAVACT 1 3 0 0", "UAVACT 1 3 0 0 0 1",
                                   "UAVACT 1 3 nan 0 0", "UAVACT 1 3 2 0 0"})
    {
        bool failed=false;
        try { std::istringstream bad(line); recovery::ReadAction(bad,3,3); }
        catch (const std::runtime_error &) { failed=true; }
        assert(failed);
    }
}
