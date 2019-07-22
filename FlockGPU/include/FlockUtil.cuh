#ifndef FLOCKUTIL_CUH
#define FLOCKUTIL_CUH
#include "FlockSystem.h"


/// A hash value that indicates the point is outside the grid
/// Code by Richard Southern from : https://github.com/NCCA/libfluid/blob/master/libfluid/

/// @brief determine grid index from position of a boid
/// @param[in] position of boid
/// @return global grid index
__device__ int3 gridFromPoint(const float3 pt);
/// @brief determines global cell index from grid index
/// @param[in] grid index of boid
/// @return global cell index
__device__ uint cellFromGrid(const uint3 grid);
/// @brief determines global cell index from grid index, but checks for out-of-bound cases
/// @param[in] grid index of grid
/// @return global cell index or NULL_HASH
__device__ uint cellFromGrid(const int3 grid);
/// @brief determines distance^2 between two positions
/// @param[in] position 1
/// @param[in] position 2
/// @return distance^2
__device__ float dist2(const float3 &_pos1, const float3 &_pos2);

/// @brief rotates a vector around Z axis
/// @param [in] _v velocity of the boid
/// @param [in] _angle angle of rotation
/// @return a vector rotated in the z-axis, angle param in radians 
__device__ float3 rotateZ(const float3 &_v, const float &_angle);



#endif // FLOCKUTIL_CUH
