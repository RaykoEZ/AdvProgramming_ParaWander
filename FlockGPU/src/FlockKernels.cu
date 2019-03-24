#include "FlockKernels.cu"

__global__ void computeAvgNeighbourPos(
    bool *_collision, 
    float3 *_target, 
    const float3 *_pos, 
    const uint *_cellOcc, 
    const uint *_scatterAddress)
{

    uint gridCellIdx = cell_from_grid(blockIdx);
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
                otherCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, 0));
                //printf("gridCellIdx=%d, otherGridCellIdx=%d\n",gridCellIdx,otherGridCellIdx);
                // Now iterate over all particles in this neighbouring cell
                for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherCellIdx]; ++threadInBlockIdx)
                {
                    // Determine the index of the neighbouring point in that cell
                    otherBoidIdx = _scatterAddress[otherCellIdx] + threadInBlockIdx;
                    float dist2 = dist2(thisPoint, points[otherPointIdx]);
                    if ((otherBoidIdx != thisBoidIdx) && (dist2 <= params.m_invRes2))
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

__global__ void genericBehaviour( float3 *_v, float3 *_col, float3 *_target, float3 *_pos, const bool *_collision, const uint *_cellOcc, const uint *_scatterAddress)
{



}

__global__ void resolveForce(float3 * _pos, float3 * _v, const float3 *_f, const uint *_cellOcc, const uint *_scatterAddress)
{


}