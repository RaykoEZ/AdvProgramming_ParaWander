#ifndef DEVICETESTKERNELS_CUH
#define DEVICETESTKERNELS_CUH
#include "FlockKernels.cuh"
#include "FlockUtil.cuh"
/// @file Functions for testing and benchmarking for devce functionalities

__global__ void callGridFromPoint(int3 *_gridIdx, float3 *_pt);
__global__ void callCellFromGrid( uint *_cellIdx, int3 *_grid);
__global__ void callDist2( float *_dist2, float3 *_pos1, float3 *_pos2);
__global__ void callRotateZ( float3 *_rot, float3 *_v,  float *_angle);
__global__ void callResolveForce(float3 *_pos, float3 *_v, const float3 *_f, const float _vMax);

__global__ void callWander( float3 *_target, float *_angle,  float3 *_v,  float3 *_pos);
__global__ void callSeek(  float3 *_f, float3  *_pos,  float3  *_v,  float3 *_target, const float _vMax);
__global__ void callFlee(  float3 *_f, float3  *_pos,  float3  *_v,  float3  *_target, const float _vMax);

 
void testGridFromPoint(thrust::device_vector<int3> &_gridIdx, thrust::device_vector<float3> &_pt);
void testCellFromGrid( thrust::device_vector<uint> &_cellIdx, thrust::device_vector<int3> &_grid);
void testDist2( thrust::device_vector<float> &_dist2, thrust::device_vector<float3> &_pos1,  thrust::device_vector<float3> &_pos2);
void testRotateZ( thrust::device_vector<float3> &_rot, thrust::device_vector<float3> &_v,  thrust::device_vector<float> &_angle);
void testResolveForce(
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_f, 
    float &_vMax);

void testWander( 
    thrust::device_vector<float3> &_target, 
    thrust::device_vector<float> &_angle, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3 >&_pos);
void testSeek( 
    thrust::device_vector<float3> &_f, 
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_target, 
    float &_vMax);
void testFlee( 
    thrust::device_vector<float3> &_f, 
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_target, 
    float &_vMax);


void testNeighbour(
    const float &_dt,
    const uint &_numP,
    const float &_res);

void testHash(
    const float &_dt,
    const uint &_numP,
    const float &_res);
#endif //DEVICETESTKERNELS_CUH
