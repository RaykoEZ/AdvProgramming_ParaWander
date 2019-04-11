#ifndef DEVICETESTKERNELS_CUH
#define DEVICETESTKERNELS_CUH
#include "FlockKernels.cuh"
#include "FlockUtil.cuh"

__global__ void callGridFromPoint(int3 *_gridIdx, float3 *_pt);
__global__ void callCellFromGrid( uint *_cellIdx, int3 *_grid);
__global__ void callDist2( float *_dist2, float3 *_pos1,  float3 *_pos2);
__global__ void callRotateZ( float3 *_rot, float3 *_v,  float *_angle);
__global__ void callResolveForce(float3 *_pos, float3 *_v, const float3 &_f, const float &_vMax);

__global__ void callWander( float3 *_target, float *_angle,  float3 *_v,  float3 *_pos);
__global__ void callSeek(  float3 *_f, float3  *_pos,  float3  *_v,  float3 *_target,  float &_vMax);
__global__ void callFlee(  float3 *_f, float3  *_pos,  float3  *_v,  float3  *_target,  float &_vMax);

#endif //DEVICETESTKERNELS_CUH
