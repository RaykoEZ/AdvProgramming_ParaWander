#include "FlockActions.cuh"
#include "FlockUtil.cuh"
#include "FlockParams.cuh"



__device__ float3 boidWanderPattern(const float3 &_target,
    const float &_angle,
    const float3 &_v,
    const float3 &_pos)
{

    /// get boid's random angle for rotation
    float thisAngle = _angle * 360.0f * CUDART_RADIAN_F;
    /// get a future direction and randomly generate possible future directions
    float3 future = _pos + (10.0f * _v);
    /// set boid target position to a random rotated direction from a position ahead of the boid
    return future + rotateZ(_v,thisAngle);
    

}


__device__ float3 boidSeekPattern( const float3  &_pos, const float3  &_v, const float3  &_target)
{

    float3 desiredV = _target;
    desiredV = desiredV - _pos;
    
        if (length(desiredV) > 0.0f)
        {
            //std::cout <<"Seeking "<<'\n';
            desiredV = normalize(desiredV);
            desiredV *= paramData.m_vMax;
            desiredV = desiredV - _v;
    
            return desiredV;
        }
        //std::cout <<"Reached " <<'\n';
    
        return desiredV;

}


__device__ float3 boidFleePattern( const float3  &_pos, float3 &_v, const float3 &_target)
{
    float3 desiredV = _target;
    desiredV = desiredV - _pos;

    if (length(desiredV)> 0.0f)
    {
        desiredV = normalize(desiredV);
        desiredV *= paramData.m_vMax;
        desiredV = desiredV - _v;
        // Draw direction line for debug

        return desiredV;
    }
    return -_v;


}
