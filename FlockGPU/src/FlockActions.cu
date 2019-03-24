#include "FlockActions.cuh"
#include "FlockUtil.cuh"

__global__ void boidWanderPattern( 
    float3 *_target,
    const float *_angle, 
    const float3 *_v, 
    const float3 *_pos, 
    const uint *_cellOcc, 
    const uint *_scatterAddress)
{
    uint gridCellIdx = cell_from_grid(blockIdx);
    
    if (threadIdx.x < _cellOcc[gridCellIdx])
    {
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPoint = _pos[thisBoidIdx];
        float3 thisV = _v[thisBoidIdx];
        /// get boid's random angle for rotation 
        float thisAngle = _angle[thisBoidIdx] * 360.0f * CUDART_RADIAN_F;
        /// get a future direction and randomly generate possible future directions
        float3 future = thisPoint + 10.0f * thisV;
        /// set boid target position to a random rotated direction from a position ahead of the boid
        _target[thisBoidIdx] = future + rotateZ(thisV,thisAngle);
    

    }
}


__device__ float3 boidSeekPattern( const float3 * _pos, const float3 * _v, const float3 *& _target, const uint *_cellOcc, const uint *_scatterAddress)
{

    float3 desiredV = _target - _pos;
    
        if (length(desiredV) > 0.0f)
        {
            //std::cout <<"Seeking "<<'\n';
            desiredV = normalize(desiredV);
            desiredV *= _vMax;
            desiredV -= _v;
    
            return desiredV;
        }
        //std::cout <<"Reached " <<'\n';
    
        return desiredV;

}


__device__ float3 boidFleePattern( const float3 * _pos, const float3 * _v, const float3 *& _target, const uint *_cellOcc, const uint *_scatterAddress)
{
    float3 desiredV =  _pos - _target;
    if (glm::length(desiredV)> 0.0f)
    {
        desiredV = normalize(desiredV);
        desiredV *= _vMax;
        desiredV -= _v;
        // Draw direction line for debug

        return desiredV;
    }
    return -_v;


}