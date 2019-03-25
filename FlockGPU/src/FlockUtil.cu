#include "FlockUtil.cuh"
#include "helper_math.h"
#include "FlockParams.cuh"
/**
 * Compute a grid position from a point. Note that the grid position can be negative in this case
 * if the point is outside [0,1] range. Sanity checks will need to be performed elsewhere.
 */
__device__ int3 gridFromPoint(const float3 pt)
{
    return make_int3(floor(pt.x * paramData.m_res), floor(pt.y * paramData.m_res), floor(pt.z * paramData.m_res));
}
/**
 * Compute a cell index from a grid. In this case we make the grid a unsigned int so no bounds checking 
 * applies.
 */
__device__ uint cellFromGrid(const uint3 grid)
{
    return grid.x + grid.y * paramData.m_res + grid.z * paramData.m_res2;
}
/**
 * Compute a cell index from a grid. In this case we make the grid an int and apply bounds checking.
 */
__device__ uint cellFromGrid(const int3 grid)
{
    // Test to see if all of the points are inside the grid (I don't think CUDA can do lazy evaluation (?))
    bool isInside = (grid.x >= 0) && (grid.x < paramData.m_res) &&
                    (grid.y >= 0) && (grid.y < paramData.m_res) &&
                    (grid.z >= 0) && (grid.z < paramData.m_res);

    // Write out the hash value if the point is within range [0,1], else write NULL_HASH
    return (isInside) ? cellFromGrid(make_uint3(grid.x, grid.y, grid.z)) : NULL_HASH;
}

__device__ uint3 gridFromCell(const uint cell)
{
    uint3 ret_val;
    ret_val.z = float(cell) * paramData.m_invRes2;
    ret_val.y = float(cell - ret_val.z * paramData.m_res2) * paramData.m_invRes;
    ret_val.x = (cell - ret_val.z * paramData.m_res2 - ret_val.y * paramData.m_res);
    return ret_val;
}
__device__ float dist2(const float3 &_pos1, const float3 &_pos2)
{
    float3 diff = _pos1 - _pos2;
    return dot(diff, diff);
}

/// _angle in radians
__device__ float3 rotateZ(const float3 &_v, const float &_angle)
{
    float3 res = _v;
    
    float Cos = (cosf(_angle));
    float Sin = (sinf(_angle));

    res.x = _v.x * Cos - _v.y * Sin;
    res.y = _v.x * Sin + _v.y * Cos;
    return res;

}
