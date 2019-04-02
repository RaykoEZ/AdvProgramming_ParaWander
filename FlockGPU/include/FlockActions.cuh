#ifndef FLOCKACTIONS_H
#define FLOCKACTIONS_H
#include <cuda_runtime.h>
#include "helper_math.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

__device__ float3 boidWanderPattern( const float &_angle, const float3 &_v, const float3 &_pos);

__device__ float3 boidSeekPattern( const float3  &_pos,  float3  &_v, const float3 &_target, const float &_vMax);

__device__ float3 boidFleePattern( const float3  &_pos,  float3  &_v, const float3  &_target, const float &_vMax);
#endif //FLOCKACTIONS_H
