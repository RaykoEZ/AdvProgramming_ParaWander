#include "FlockUtil.cuh"

/**
 * Compute a grid position from a point. Note that the grid position can be negative in this case
 * if the point is outside [0,1] range. Sanity checks will need to be performed elsewhere.
 */
__device__ int3 gridFromPoint(const float3 pt)
{
    return make_int3(floor(pt.x * params.m_res), floor(pt.y * params.m_res), floor(pt.z * params.m_res));
}
/**
 * Compute a cell index from a grid. In this case we make the grid a unsigned int so no bounds checking 
 * applies.
 */
__device__ uint cellFromgrid(const uint3 grid)
{
    return grid.x + grid.y * params.m_res + grid.z * params.m_res2;
}
/**
 * Compute a cell index from a grid. In this case we make the grid an int and apply bounds checking.
 */
__device__ uint cellFromGrid(const int3 grid)
{
    // Test to see if all of the points are inside the grid (I don't think CUDA can do lazy evaluation (?))
    bool isInside = (grid.x >= 0) && (grid.x < params.m_res) &&
                    (grid.y >= 0) && (grid.y < params.m_res) &&
                    (grid.z >= 0) && (grid.z < params.m_res);

    // Write out the hash value if the point is within range [0,1], else write NULL_HASH
    return (isInside) ? cellFromgrid(make_uint3(grid.x, grid.y, grid.z)) : NULL_HASH;
}

__device__ uint3 gridFromCell(const uint cell)
{
    uint3 ret_val;
    ret_val.z = float(cell) * params.m_h2;
    ret_val.y = float(cell - ret_val.z * params.m_res2) * params.m_h;
    ret_val.x = (cell - ret_val.z * params.m_res2 - ret_val.y * params.m_res);
    return ret_val;
}