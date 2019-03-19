// For the CUDA runtime routines (prefixed with "cuda_")
#include "Hash.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>


// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

/// Point hashing operator to calculate a position's cell index, referenced from Richard Southern's libfluid sample

PointHashOperator::PointHashOperator(uint *_occ)
    : m_cellOcc(_occ)
{}

/// The operator functor. Should be called with thrust::transform and a zip_iterator
__device__ uint PointHashOperator::operator()(const float3 &_pos)
{
    // Note that finding the grid coordinates are much simpler if the grid is over the range [0,1] in
    // each dimension and the points are also in the same space.
    //int3 grid = grid_from_point(_pos);

    // Compute the hash for this grid cell
    //uint hash = cell_from_grid(grid);

    // Calculate the cell occupancy counter here to save on an extra kernel launch (won't trigger if out of bounds)
    //if (hash != NULL_HASH)
    //{
    //    atomicAdd(&m_cellOcc[hash], 1);
    //}

    // Return the cell idx (NULL_HASH if out of bounds)
    return 0; //hash;
}