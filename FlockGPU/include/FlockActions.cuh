#ifndef FLOCKACTIONS_H
#define FLOCKACTIONS_H
#include <cuda_runtime.h>
#include "helper_math.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

/// @brief handles wander pattern for a "boid"
/// @param [in] _angle angle to steer
/// @param [in] _v velocity of a boid
/// @param [in] _pos postion of a boid
/// @return target position to steer for wandering pattern
__device__ float3 boidWanderPattern( const float &_angle, const float3 &_v, const float3 &_pos);

/// @brief handles seek pattern for a "boid"
/// @param [in] _pos postion of a boid
/// @param [in] _v velocity of a boid
/// @param [in] _target target position of a boid
/// @param [in] _vMax limit velocity of a boid
/// @return
__device__ float3 boidSeekPattern( const float3  &_pos,  float3  &_v, const float3 &_target, const float &_vMax);

/// @brief handles flee pattern for a "boid"
/// @param [in] _pos postion of a boid
/// @param [in] _v velocity of a boid
/// @param [in] _target target position of a boid
/// @param [in] _vMax velocity limit of a boid
/// @return
__device__ float3 boidFleePattern( const float3  &_pos,  float3  &_v, const float3  &_target, const float &_vMax);
#endif //FLOCKACTIONS_H
