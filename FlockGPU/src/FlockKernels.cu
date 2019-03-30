#include "FlockKernels.cuh"
#include "FlockActions.cuh"
#include "FlockParams.cuh"
#include "FlockUtil.cuh"
__global__ void computeAvgNeighbourPos(
    bool *_collision, 
    float3 *_target, 
    const float3 *_pos, 
    const uint *_cellOcc, 
    const uint *_scatterAddress)
{

    uint gridCellIdx = cellFromGrid(blockIdx);
    uint numNeighbour = 0;
    float3 sumPos = make_float3(0.0f,0.0f,0.0f);
    /// 
    if (threadIdx.x <_cellOcc[gridCellIdx])
    {
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPoint = _pos[thisBoidIdx];
        // set collision flag to false here
        _collision[thisBoidIdx] = false;

        int threadInBlockIdx;
        uint otherCellIdx;
        uint otherBoidIdx;
        /// index through self and neighbours around a cell
        for (int i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (int j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                
                // Calculate the index of the other grid cell
                otherCellIdx = cellFromGrid(make_uint3(blockIdx.x + i, blockIdx.y + j, 0));
                //printf("gridCellIdx=%d, otherGridCellIdx=%d\n",gridCellIdx,otherGridCellIdx);
                // Now iterate over all particles in this neighbouring cell
                for (threadInBlockIdx = 0; threadInBlockIdx < _cellOcc[otherCellIdx]; ++threadInBlockIdx)
                {
                    // Determine the index of the neighbouring point in that cell
                    otherBoidIdx = _scatterAddress[otherCellIdx] + threadInBlockIdx;
                    float d2 = dist2(thisPoint, _pos[otherBoidIdx]);
                    if ((otherBoidIdx != thisBoidIdx) && (d2 <= paramData.m_collisionRad))
                    {
                         /// sum position to prepare for average position
                         _collision[thisBoidIdx] = true;
                        sumPos = make_float3(sumPos.x + _pos[otherBoidIdx].x,sumPos.y + _pos[otherBoidIdx].y,0.0f);
                        ++numNeighbour;
                    }          
                }
            }
        }
        if(numNeighbour > 0)
        {
            /// set average position
            _collision[thisBoidIdx] = true;
            _target[thisBoidIdx] = make_float3(sumPos.x / numNeighbour, sumPos.y / numNeighbour,0.0f);
        }
        else
        {

            _collision[thisBoidIdx] = false;
        }
    }
}

__global__ void genericBehaviour( 
    float3 *_v, 
    float3 *_col, 
    float3 *_target, 
    float3 *_pos, 
    bool *_collision, 
    const uint *_cellOcc, 
    const uint *_scatterAddress,
    const float *_angle,
    const float * _vMax)
{

    uint gridCellIdx = cellFromGrid(blockIdx);
    ///

    if (threadIdx.x < _cellOcc[gridCellIdx])
    {       
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPos = _pos[thisBoidIdx];
        float3 thisV = _v[thisBoidIdx];
        float3 f = make_float3(0.0f,0.0f,0.0f);
        float3 thisTarget = _target[thisBoidIdx];
        float thisAng = _angle[thisBoidIdx];
        float thisVMax = _vMax[thisBoidIdx];

        if(_collision[thisBoidIdx])
        {
            f = boidFleePattern(thisPos,
                                thisV,
                                thisTarget,
                                thisVMax);

            _col[thisBoidIdx] = make_float3(255.0f,0.0f,0.0f);

        }
        else
        {
            _target[thisBoidIdx] = boidWanderPattern(
                        thisAng,
                        thisV,
                        thisPos);

            f = boidSeekPattern(
                        thisPos,
                        thisV,
                        thisTarget,
                        thisVMax);

            _col[thisBoidIdx] = make_float3(0.0f,255.0f,0.0f);

        }


        resolveForce(thisPos,thisV,f,thisVMax);

        _pos[thisBoidIdx] = thisPos;
        _v[thisBoidIdx] = thisV;

    }
    else
    {
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        _col[thisBoidIdx] = make_float3(0.0f,0.0f,0.0f);
    }
}

__device__ void resolveForce(
    float3 &_pos,
    float3 &_v,
    const float3 &_f,
    const float &_vMax)
{
   

    float3 accel = make_float3(_f.x * paramData.m_invMass,_f.y * paramData.m_invMass,0.0f);
    float3 oldV = make_float3(_v.x + accel.x,_v.y + accel.y,0.0f);

    if(length(_v) > 0.0f)
    {
        _v = clamp(oldV,
                make_float3(-_vMax,-_vMax,0.0f),
                make_float3(_vMax,_vMax,0.0f));
        _v = normalize(_v);

    }
    
    float3 pos = make_float3(_pos.x +_v.x * paramData.m_dt ,_pos.y + _v.y * paramData.m_dt, 0.0f);

    _pos = pos;

}
