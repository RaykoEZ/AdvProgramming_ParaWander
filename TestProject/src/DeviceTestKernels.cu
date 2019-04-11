#include "DeviceTestKernels.cuh"

__global__ void callGridFromPoint( int3 *_gridIdx, float3 *_pt)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _gridIdx[gIdx] = gridFromPoint(_pt[gIdx]);
}
__global__ void callCellFromGrid( uint *_cellIdx, int3 *_grid)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;

    _cellIdx[gIdx] = cellFromGrid(_grid[gIdx]);
}
__global__ void callDist2( float *_dist2, float3 *_pos1,  float3 *_pos2)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _dist2[gIdx] = dist2(_pos1[gIdx],_pos2[gIdx]);
}
__global__ void callRotateZ( float3 *_rot, float3 *_v,  float *_angle)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _rot[gIdx] = rotateZ(_v[gIdx],_angle[gIdx]);

}

__global__ void callResolveForce(float3 *_pos, float3 *_v, const float3 &_f, const float &_vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    callResolveForce(_pos[gIdx], _v[gIdx], _f[gIdx], _vMax);
}


__global__ void callWander( float3 *_target, float *_angle,  float3 *_v,  float3 *_pos)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _target[gIdx] = boidWanderPattern(_angle[gIdx], _v[gIdx], _pos[gIdx]);
}

__global__ void callSeek(  float3 *_f, float3  *_pos,  float3  *_v,  float3 *_target, const float &_vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _f[gIdx] = boidSeekPattern(_pos[gIdx], _v[gIdx], _target[gIdx], _vMax);
}

__global__ void callFlee(  float3 *_f, float3  *_pos,  float3  *_v,  float3  *_target, const float &_vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _f[gIdx] = boidFleePattern(_pos[gIdx], _v[gIdx], _target[gIdx], _vMax);
}
