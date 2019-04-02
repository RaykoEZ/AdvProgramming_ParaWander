#include "FlockActions.cuh"
#include "FlockUtil.cuh"
#include "FlockParams.cuh"



__device__ float3 boidWanderPattern(
    const float &_angle,
    const float3 &_v,
    const float3 &_pos)
{

    /// get boid's random angle for rotation
    float thisAngle = _angle * 360.0f * RADIANS_F;
    /// get a future direction and randomly generate possible future directions
    float3 future = make_float3(_pos.x + 10.0f * _v.x, _pos.y +  10.0f *_v.y,0.0f); 
    /// set boid target position to a random rotated direction from a position ahead of the boid
    float3 rot = rotateZ(_v,thisAngle);
    return future + rot;
    

}


__device__ float3 boidSeekPattern( const float3  &_pos,  float3  &_v, const float3  &_target, const float &_vMax)
{

    float3 desiredV = _target - _pos;
    
        if (length(desiredV) > 0.0f)
        {
            //std::cout <<"Seeking "<<'\n';
            desiredV = normalize(desiredV);
            desiredV = desiredV * _vMax;
            desiredV = desiredV - _v;
            _v = desiredV;
            return desiredV;
        }
        //std::cout <<"Reached " <<'\n';
    
        return _v;

}


__device__ float3 boidFleePattern( const float3  &_pos, float3 &_v, const float3 &_target, const float &_vMax)
{
    float3 desiredV = _pos - _target;

    if (length(desiredV)> 0.0f)
    {
        //std::cout <<"Seeking "<<'\n';
        desiredV = normalize(desiredV);
        desiredV = desiredV * _vMax;
        desiredV = desiredV - _v;
        _v = desiredV;


        return desiredV;
    }
    return _v;


}
