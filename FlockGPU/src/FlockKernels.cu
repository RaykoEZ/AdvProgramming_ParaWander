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
    if (threadIdx.x < _cellOcc[gridCellIdx])
    {
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPoint = _pos[thisBoidIdx];

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
                    if ((otherBoidIdx != thisBoidIdx) && (d2 <= paramData.m_invRes2))
                    {
                         /// sum position to prepare for average position
                        sumPos += _pos[otherBoidIdx];
                        ++numNeighbour;
                    }          
                }
            }
        }
        if(numNeighbour > 0)
        {
            /// set average position
            _collision[thisBoidIdx] = true;
            _target[thisBoidIdx] = sumPos / numNeighbour;
        }
    }
}

__global__ void genericBehaviour( 
    float3 *_v, 
    float3 *_col, 
    float3 *_target, 
    float3 *_pos, 
    const bool *_collision, 
    const uint *_cellOcc, 
    const uint *_scatterAddress,
    const float *_angle)
{

    uint gridCellIdx = cellFromGrid(blockIdx);
    //float3 f;
    ///
    if (threadIdx.x < _cellOcc[gridCellIdx])
    {
        uint thisBoidIdx = _scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPos = _pos[thisBoidIdx];
        float3 thisV = _v[thisBoidIdx];
        float3 f;
        float3 thisTarget = _target[thisBoidIdx];
        float thisAng = _angle[thisBoidIdx];

        if(_collision[thisBoidIdx])
        {
            f = boidFleePattern(thisPos,
                                thisV,
                                thisTarget);

            _col[thisBoidIdx] = make_float3(255.0f,0.0f,0.0f);

        }
        else
        {
            _target[thisBoidIdx] = boidWanderPattern(
                        thisTarget,
                        thisAng,
                        thisV,
                        thisPos);

            f = boidSeekPattern(
                        thisPos,
                        thisV,
                        thisTarget);

            _col[thisBoidIdx] = make_float3(0.0f,255.0f,0.0f);

        }

        resolveForce(thisPos, thisV, f);
    }
}

__device__ void resolveForce(
    float3 &_pos,
    float3 &_v,
    const float3 &_f)
{

    float3 accel = _f * paramData.m_invMass;
    float3 oldV = _v + accel;

    if(length(_v) > 0.0f)
    {

         _v = clamp(oldV,
              make_float3(-paramData.m_vMax,-paramData.m_vMax,0.0f),
              make_float3(paramData.m_vMax,paramData.m_vMax,0.0f));
         _v = normalize(_v);

    }
    _pos += _v * paramData.m_dt * paramData.m_invRes2;



}
