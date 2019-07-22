#ifndef FLOCKKERNELS_CUH
#define FLOCKKERNELS_CUH
#include "FlockActions.cuh"

/// @brief calculates an average neighbourhood position for each thread/boid
/// @param [in/out] _collision collision flag to be determined
/// @param [in/out] _target target positions of all boids
/// @param [in] _pos position of all boids
/// @param [in] _cellOcc cellOccupancy for the world
/// @param [in] _scatterAddress scatterAddresses to index into each sorted boid data
__global__ void computeAvgNeighbourPos(bool *_collision, float3 *_target, const float3 *_pos, const uint *_cellOcc, const uint *_scatterAddress);

/// @brief handles behaviour for boids with a simple state machine
/// @param [in/out] _v velocity of all boids
/// @param [in/out] _col colour for all boids to be determined
/// @param [in/out] _target target position of all boids
/// @param [in/out] _pos position of all boids
/// @param [in/out] _collision collision flag of all boids to be read and reset
/// @param [in] _cellOcc cell occupancy for the world
/// @param [in] _scatterAddress scatterAddresses to index into each sorted boid data
/// @param [in] _angle all random angles to steer if a boid is wandering
/// @param [in] _vMax velocity limit of all boids
__global__ void genericBehaviour(
    float3 *_v, 
    float3 *_col, 
    float3 *_target, 
    float3 *_pos, 
    bool *_collision, 
    const uint *_cellOcc, 
    const uint *_scatterAddress, 
    float *_angle, 
    const float * _vMax /*, uint * _threadIdx, uint * _blockIdx*/);

/// @brief updates a boid with the forces given
/// @param [in/out] _pos position of a boid
/// @param [in/out] _v velocity vector of a boid
/// @param [in] _f resultant force of a boid
/// @param [in] _vMax velocity limit of a boid
__device__ void resolveForce(float3 &_pos, float3 &_v, const float3 &_f, const float &_vMax);

#endif // FLOCKKERNELS_CUH
