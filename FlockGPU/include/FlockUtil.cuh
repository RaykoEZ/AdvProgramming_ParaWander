#ifndef FLOCKUTIL_CUH
#define FLOCKUTIL_CUH
#include "FlockSystem.h"


/// A hash value that indicates the point is outside the grid
/// Code by Richard Southern from : https://github.com/NCCA/libfluid/blob/master/libfluid/
__device__ int3 gridFromPoint(const float3 pt);
__device__ uint cellFromGrid(const uint3 grid);
__device__ uint cellFromGrid(const int3 grid);
__device__ uint3 gridFromCell(const uint cell);
__device__ float dist2(const float3 &_pos1, const float3 &_pos2);

/// returns a vector rotated in the z-axis, angle param in radians 
__device__ float3 rotateZ(const float3 &_v, const float &_angle);



#endif // FLOCKUTIL_CUH
