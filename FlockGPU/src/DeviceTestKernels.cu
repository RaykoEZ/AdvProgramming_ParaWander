#include "DeviceTestKernels.cuh"
#include <iostream>


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
    _rot[gIdx] = rotateZ(_v[gIdx],_angle[gIdx] * 360.0f * RADIANS_F);

}

__global__ void callResolveForce(float3 *_pos, float3 *_v, const float3 *_f, const float _vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    //resolveForce(_pos[gIdx], _v[gIdx], _f[gIdx], _vMax);
    resolveForce(_pos[gIdx], _v[gIdx], _f[gIdx], _vMax);
}


__global__ void callWander( float3 *_target, float *_angle,  float3 *_v,  float3 *_pos)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _target[gIdx] = boidWanderPattern(_angle[gIdx], _v[gIdx], _pos[gIdx]);
}

__global__ void callSeek(  float3 *_f, float3  *_pos,  float3  *_v,  float3 *_target, const float _vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _f[gIdx] = boidSeekPattern(_pos[gIdx], _v[gIdx], _target[gIdx], _vMax);
}

__global__ void callFlee(  float3 *_f, float3  *_pos,  float3  *_v,  float3  *_target, const float _vMax)
{
    uint gIdx = blockIdx.x *blockDim.x + threadIdx.x;
    _f[gIdx] = boidFleePattern(_pos[gIdx], _v[gIdx], _target[gIdx], _vMax);
}




void testGridFromPoint(thrust::device_vector<int3> &_gridIdx, thrust::device_vector<float3> &_pt)
{
    int3 * grid = thrust::raw_pointer_cast(&_gridIdx[0]);
    float3 * pos = thrust::raw_pointer_cast(&_pt[0]);

    callGridFromPoint<<<1, _gridIdx.size()>>>(grid, pos);
    cudaThreadSynchronize();
}
void testCellFromGrid( thrust::device_vector<uint> &_cellIdx, thrust::device_vector<int3> &_grid)
{
    uint * cell = thrust::raw_pointer_cast(&_cellIdx[0]);
    int3 * grid = thrust::raw_pointer_cast(&_grid[0]);
    callCellFromGrid<<<1, _cellIdx.size()>>>(cell, grid);
    cudaThreadSynchronize();
}
void testDist2( thrust::device_vector<float> &_dist2, thrust::device_vector<float3> &_pos1,  thrust::device_vector<float3> &_pos2)
{
    float * dist2 = thrust::raw_pointer_cast(&_dist2[0]);
    float3 * pos1 = thrust::raw_pointer_cast(&_pos1[0]);
    float3 * pos2 = thrust::raw_pointer_cast(&_pos2[0]);
    callDist2<<<1, _dist2.size()>>>(dist2, pos1, pos2);
    cudaThreadSynchronize();
}
void testRotateZ( thrust::device_vector<float3> &_rot, thrust::device_vector<float3> &_v,  thrust::device_vector<float> &_angle)
{
    float3 * rot = thrust::raw_pointer_cast(&_rot[0]);
    float3 * v = thrust::raw_pointer_cast(&_v[0]);
    float * angle = thrust::raw_pointer_cast(&_angle[0]);
    callRotateZ<<<1, _rot.size()>>>(rot, v, angle);
    cudaThreadSynchronize();
}
void testResolveForce(
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_f, 
    float &_vMax)
{
    float3 * pos = thrust::raw_pointer_cast(&_pos[0]);
    float3 * v = thrust::raw_pointer_cast(&_v[0]);
    float3 * f = thrust::raw_pointer_cast(&_f[0]);

    thrust::device_ptr<float> vMax(&_vMax);
    std::cout<< "Test case count: " << *vMax.get() <<'\n';

    callResolveForce<<<1, _pos.size()>>>( pos, v, f, *vMax.get());
    cudaThreadSynchronize();

}

void testWander( 
    thrust::device_vector<float3> &_target, 
    thrust::device_vector<float> &_angle, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3 >&_pos)
{
    float3 * target= thrust::raw_pointer_cast(&_target[0]);
    float * angle = thrust::raw_pointer_cast(&_angle[0]);
    float3 * v = thrust::raw_pointer_cast(&_v[0]);
    float3 * pos = thrust::raw_pointer_cast(&_pos[0]);
    callWander<<<1, _target.size()>>>( target, angle, v, pos);
    cudaThreadSynchronize();
}
void testSeek( 
    thrust::device_vector<float3> &_f, 
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_target,  
    float &_vMax)
{
    float3 * f = thrust::raw_pointer_cast(&_f[0]);
    float3 * pos = thrust::raw_pointer_cast(&_pos[0]);
    float3 * v = thrust::raw_pointer_cast(&_v[0]);
    float3 * target = thrust::raw_pointer_cast(&_target[0]);

    thrust::device_ptr<float> vMax(&_vMax);
    //float * vMaxPtr = thrust::raw_pointer_cast(vMax.get());
    callSeek<<<1, _f.size()>>>(f, pos, v, target, *vMax.get());
    cudaThreadSynchronize();
}
void testFlee( 
    thrust::device_vector<float3> &_f, 
    thrust::device_vector<float3> &_pos, 
    thrust::device_vector<float3> &_v, 
    thrust::device_vector<float3> &_target, 
    float &_vMax)
{
    float3 * f = thrust::raw_pointer_cast(&_f[0]);
    float3 * pos = thrust::raw_pointer_cast(&_pos[0]);
    float3 * v = thrust::raw_pointer_cast(&_v[0]);
    float3 * target = thrust::raw_pointer_cast(&_target[0]);

    thrust::device_ptr<float> vMax(&_vMax);
    //float * vMaxPtr = thrust::raw_pointer_cast(vMax.get());
    callFlee<<<1, _f.size()>>>(f, pos, v, target, *vMax.get());
    cudaThreadSynchronize();
}