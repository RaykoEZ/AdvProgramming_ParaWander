#include "DeviceTestKernels.cuh"

__host__ int3 callGridFromPoint(const float3 _pt)
{
    int3 ret = make_int3(0,0,0);
    return ret;
}
__host__ uint callCellFromGrid(const int3 _grid)
{
    uint ret = 0;
    return ret;
}
__host__ float callDist2(const float3 &_pos1, const float3 &_pos2)
{
    float ret = 0.0f;
    return ret;
}
__host__ float3 callRotateZ(const float3 &_v, const float &_angle)
{
    float3 ret = make_float3(0.0f,0.0f,0.0f);
    return ret;
}

__host__ void callResolveForce(float3 &_pos, float3 &_v, const float3 &_f, const float &_vMax)
{

}


__host__ float3 callWander(const float &_angle, const float3 &_v, const float3 &_pos)
{
    float3 ret = make_float3(0.0f,0.0f,0.0f);
    return ret;
}

__host__ float3 callSeek( const float3  &_pos,  float3  &_v, const float3 &_target, const float &_vMax)
{
    float3 ret = make_float3(0.0f,0.0f,0.0f);
    return ret;
}

__host__ float3 callFlee( const float3  &_pos,  float3  &_v, const float3  &_target, const float &_vMax)
{
    float3 ret = make_float3(0.0f,0.0f,0.0f);
    return ret;
}
