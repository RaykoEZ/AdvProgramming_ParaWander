#ifndef HASH_CUH
#define HASH_CUH
#include "FlockParams.cuh"

struct PointHashOperator
{
    uint *m_cellOcc;

    PointHashOperator(uint *_occ);

    __device__ uint operator()(const float3 &_pos);
};






#endif //HASH_CUH