#include "DebugUtil.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>


void cudaErrorPrint()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Thrust allocation failed, error " << cudaGetErrorString(err) << "\n";
        
    }

}