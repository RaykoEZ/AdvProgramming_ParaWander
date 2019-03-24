#include "FlockActions.cuh"




__global__ void boidWanderPattern( 
    float3 *_target, 
    const float3 *_v, 
    const float3 *_pos, 
    const uint *_cellOcc, 
    const uint *_scatterAddress)
{


}


__global__ float3 boidSeekPattern( const float3 * _pos, const float3 * _v, const uint *_cellOcc, const uint *_scatterAddress)
{



}


__global__ float3 boidFleePattern( const float3 * _pos, const float3 * _v, const uint *_cellOcc, const uint *_scatterAddress)
{



}