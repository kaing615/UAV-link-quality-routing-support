#define UAV_SERVICE_FILTER_ONLY
#include "../../simulation/ns3/service-observation.h"
#include <cassert>
int main()
{
    // Bound source client accepts its wildcard local socket; destination is
    // exact. Bound destination server accepts wildcard local IP; source exact.
    assert(ServiceFlowAddressMatch(0,2,12345,9000,1,2,0,true,false));
    assert(ServiceFlowAddressMatch(1,0,12345,9000,1,2,12345,false,true));
    assert(ServiceFlowAddressMatch(1,2,12345,9000,1,2,12345,false,true));
    assert(!ServiceFlowAddressMatch(3,0,12345,9000,1,2,12345,false,true));
    assert(!ServiceFlowAddressMatch(1,0,12346,9000,1,2,12345,false,true));
    assert(!ServiceFlowAddressMatch(1,0,12345,9001,1,2,12345,false,true));
    assert(!ServiceFlowAddressMatch(0,3,12345,9000,1,2,0,true,false));
    assert(!ServiceFlowAddressMatch(1,2,0,9000,1,2,0,true,false));
}
