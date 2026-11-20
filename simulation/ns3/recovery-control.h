#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace recovery
{
struct Vec { double x = 0, y = 0, z = 0; };
struct Bounds { double xMin, xMax, yMin, yMax, zMin, zMax; };
inline Vec Sub(Vec a, Vec b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline double Dot(Vec a, Vec b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline double Distance(Vec a, Vec b) { Vec d = Sub(a,b); return std::sqrt(Dot(d,d)); }
inline bool Finite(Vec a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

inline double SweptSeparation(Vec a, Vec nextA, Vec b, Vec nextB)
{
    Vec r = Sub(a,b), v = Sub(Sub(nextA,a), Sub(nextB,b));
    double vv = Dot(v,v);
    double t = vv > 0 ? std::clamp(-Dot(r,v)/vv, 0.0, 1.0) : 0.0;
    return Distance({r.x+t*v.x, r.y+t*v.y, r.z+t*v.z}, {});
}

struct MotionResult { std::vector<Vec> next; bool rejected = false; };

// Actions are normalized vectors; limit their Euclidean norm, not each axis's
// resulting speed. One command spans exactly one simulator second.
inline MotionResult LimitMotion(const std::vector<Vec> &positions,
                                const std::vector<Vec> &actions,
                                const std::vector<bool> &fixed,
                                const std::vector<double> &travel,
                                Bounds bounds, double speed, double budget, double minSeparation)
{
    const size_t n = positions.size();
    if (actions.size()!=n || fixed.size()!=n || travel.size()!=n ||
        !std::isfinite(speed) || speed<=0 || !std::isfinite(budget) || budget<0 ||
        !std::isfinite(minSeparation) || minSeparation<=0)
        throw std::invalid_argument("Invalid motion shape or limits");
    MotionResult result{positions};
    for (size_t i=0; i<n; ++i)
    {
        if (!Finite(positions[i]) || !Finite(actions[i]) || !std::isfinite(travel[i]) || travel[i]<0)
            throw std::invalid_argument("Nonfinite motion input");
        if (fixed[i]) { continue; }
        const Vec a = actions[i];
        const double norm = std::sqrt(Dot(a,a));
        const double scale = std::min(speed, std::max(0.0,budget-travel[i])) / std::max(1.0,norm);
        result.next[i] = {std::clamp(positions[i].x+scale*a.x,bounds.xMin,bounds.xMax),
                          std::clamp(positions[i].y+scale*a.y,bounds.yMin,bounds.yMax),
                          std::clamp(positions[i].z+scale*a.z,bounds.zMin,bounds.zMax)};
    }
    for (size_t i=0; i<n; ++i)
    {
        for (size_t j=i+1; j<n; ++j)
        {
            if (SweptSeparation(positions[i],result.next[i],positions[j],result.next[j]) < minSeparation-1e-8)
            {
                result.next=positions;
                result.rejected=true;
                return result;
            }
        }
    }
    return result;
}
} // namespace recovery
