#include "recovery-control.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    using namespace recovery;
    Bounds bounds{0, 800, 0, 800, 50, 150};
    std::vector<Vec> pos{{20, 20, 100}, {100, 100, 100}, {200, 100, 100}};
    std::vector<Vec> actions{{1, 1, 1}, {1, 1, 1}, {1, 0, 0}};
    auto move = LimitMotion(pos, actions, {true, false, false}, {0, 0, 398}, bounds, 8, 400, 5);
    assert(Distance(move.next[0], pos[0]) == 0);
    assert(std::abs(Distance(move.next[1], pos[1]) - 8) < 1e-8);
    assert(std::abs(Distance(move.next[2], pos[2]) - 2) < 1e-8);
    assert(!move.rejected);
    auto edge = LimitMotion({{799, 100, 100}}, {{1, 0, 0}}, {false}, {0}, bounds, 8, 400, 5);
    assert(edge.next[0].x == 800);
    auto depleted = LimitMotion({{100, 100, 100}}, {{1, 0, 0}}, {false}, {400}, bounds, 8, 400, 5);
    assert(depleted.next[0].x == 100);
    // Safe end points cannot detect two moving nodes crossing between snapshots.
    auto crossing = LimitMotion({{100, 100, 100}, {110, 100, 100}},
                                {{1, 0, 0}, {-1, 0, 0}}, {false, false}, {0, 0}, bounds, 10, 400, 5);
    assert(crossing.rejected);
    assert(crossing.next[0].x == 100 && crossing.next[1].x == 110);
    bool invalid = false;
    try { LimitMotion(pos, {{NAN, 0, 0}}, {false}, {0}, bounds, 8, 400, 5); }
    catch (const std::invalid_argument &) { invalid = true; }
    assert(invalid);
    std::cout << "motion constraints passed\n";
}
