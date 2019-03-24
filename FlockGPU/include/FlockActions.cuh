#ifndef FLOCKACTIONS_H
#define FLOCKACTIONS_H
#include "helper_math.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void boidWanderPattern( float3 *_target, const float *_angle, const float3 * _v, const float3 *_pos, const uint * _cellOcc, const uint * _scatterAddress);

__device__ float3 boidSeekPattern( const float3 * _pos, const float3 * _v, const float3 * _target, const uint * _cellOcc, const uint * _scatterAddress);

__device__ float3 boidFleePattern( const float3 * _pos, const float3 * _v, const float3 * _target, const uint *_cellOcc, const uint *_scatterAddress);
#endif //FLOCKACTIONS_H
