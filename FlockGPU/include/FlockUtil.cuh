#ifndef FLOCKUTIL_CUH
#define FLOCKUTIL_CUH
#include "fluidsystem.h"


/// A hash value that indicates the point is outside the grid
/// Code by Richard Southern from : https://github.com/NCCA/libfluid/blob/master/libfluid/
__device__ int3 gridFromPoint(const float3 pt);
__device__ uint cellFromgrid(const uint3 grid);
__device__ uint cellFromGrid(const int3 grid);
__device__ uint3 gridFromCell(const uint cell);



#endif // FLOCKUTIL_CUH