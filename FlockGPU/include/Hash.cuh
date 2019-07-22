#ifndef HASH_CUH
#define HASH_CUH
#include "FlockParams.cuh"

/// @brief Point hashing functor for using thrust algorithms
struct PointHashOperator
{
    ///@brief cell occupation table
    uint *m_cellOcc;

    ///@brief ctor
    PointHashOperator(uint *_occ);
    ///@brief operator for hashing
    ///@param [in] _pos position of a boid
    ///@return hash value of the position
    __device__ uint operator()(const float3 &_pos);
};






#endif //HASH_CUH