#include "FlockActions.cuh"
#include "FlockUtil.cuh"
#include "FlockParams.cuh"



__device__ float3 boidWanderPattern(
    const float &_angle,
    const float3 &_v,
    const float3 &_pos)
{

    /// get boid's random angle for rotation
    float thisAngle = (_angle * 360.0f * CUDART_RADIAN_F);
    /// get a future direction and randomly generate possible future directions
    float3 future = make_float3(_pos.x + 10.0f*_v.x, _pos.y + 10.0f*_v.y,0.0f); 
    /// set boid target position to a random rotated direction from a position ahead of the boid
    float3 rot = rotateZ(_v,thisAngle);
    return make_float3(future.x + rot.x, future.y + rot.y, 0.0f);
    

}


__device__ float3 boidSeekPattern( const float3  &_pos, const float3  &_v, const float3  &_target, const float &_vMax)
{

    float3 desiredV = _target;
    desiredV = make_float3(desiredV.x - _pos.x,desiredV.y - _pos.y,0.0f);
    
        if (length(desiredV) > 0.0f)
        {
            //std::cout <<"Seeking "<<'\n';
            desiredV = normalize(desiredV);
            desiredV = make_float3(desiredV.x * _vMax,desiredV.y * _vMax,0.0f);
            desiredV = make_float3(desiredV.x - _v.x,desiredV.y - _v.y,0.0f);;
    
            return desiredV;
        }
        //std::cout <<"Reached " <<'\n';
    
        return _v;

}


__device__ float3 boidFleePattern( const float3  &_pos, float3 &_v, const float3 &_target, const float &_vMax)
{
    float3 desiredV = _target;
    desiredV =  make_float3(_pos.x - desiredV.x,_pos.y - desiredV.y,0.0f);

    if (length(desiredV)> 0.0f)
    {
        desiredV = normalize(desiredV);
        desiredV = make_float3(desiredV.x * _vMax,desiredV.y * _vMax,0.0f);
        desiredV = make_float3(desiredV.x - _v.x,desiredV.y - _v.y,0.0f);;
        // Draw direction line for debug

        return desiredV;
    }
    return -_v;


}
