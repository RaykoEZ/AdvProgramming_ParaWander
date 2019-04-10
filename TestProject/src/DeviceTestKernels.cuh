#ifndef DEVICETESTKERNELS_CUH
#define DEVICETESTKERNELS_CUH
#include "FlockKernels.cuh"
#include "FlockUtil.cuh"

__host__ int3 callGridFromPoint(const float3 _pt);
__host__ uint callCellFromGrid(const int3 _grid);
__host__ float callDist2(const float3 &_pos1, const float3 &_pos2);
__host__ float3 callRotateZ(const float3 &_v, const float &_angle);
__host__ void callResolveForce(float3 &_pos, float3 &_v, const float3 &_f, const float &_vMax);

__host__ float3 callWander(const float &_angle, const float3 &_v, const float3 &_pos);
__host__ float3 callSeek( const float3  &_pos,  float3  &_v, const float3 &_target, const float &_vMax);
__host__ float3 callFlee( const float3  &_pos,  float3  &_v, const float3  &_target, const float &_vMax);

#endif //DEVICETESTKERNELS_CUH
