#ifndef FLOCKKERNELS_CUH
#define FLOCKKERNELS_CUH
#include "FlockActions.cuh"
/// @brief calculates an average neighbourhood position for each thread/boid
/// does collision detection

__global__ void computeAvgNeighbourPos(bool *_collision, float3 *_target, const float3 *_pos, const uint *_cellOcc, const uint *_scatterAddress);

__global__ void genericBehaviour(float3 *_v, float3 *_col, float3 *_target, float3 *_pos, const bool *_collision, const uint *_cellOcc, const uint *_scatterAddress, const float *_angle);

__device__ void resolveForce(float3 &_pos, float3 &_v, const float3 &_f);

#endif // FLOCKKERNELS_CUH
