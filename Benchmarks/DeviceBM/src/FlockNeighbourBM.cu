#include <benchmark/benchmark.h>
#include "FlockKernels.cuh"
#include "Hash.cuh"
#include "FlockSystem.h"
#include <vector>

namespace
{
    static void benchNeighbour(
        const float &_dt.  
        const uint &_numP,
        const float &_res, 
        benchmarch::State& state)
    {
        FlockSystem flockSys(_numP,10.0f,0.1f,_dt,_res);
        flockSys.init();
        thrust::device_vector<uint> d_cellOcc = flockSys.getCellOcc();
        thrust::device_vector<uint> d_scatter = flockSys.getScatterAddress();
        thrust::device_vector<uint> d_hash = flockSys.getHash();
        thrust::device_vector<bool> d_collision = flockSys.getCollisionFlag();
        thrust::device_vector<float3> d_pos = flockSys.getPos();
        thrust::device_vector<float3> d_target = flockSys.getTarget();

        float3 * pos = thrust::raw_pointer_cast(&d_pos[0]);
        float3 * targetPos = thrust::raw_pointer_cast(&d_target[0]);
        bool * collision = thrust::raw_pointer_cast(&d_collision[0]);
        uint * cellOcc = thrust::raw_pointer_cast(&d_cellOcc[0]);
        uint * scatter = thrust::raw_pointer_cast(&d_scatter[0]);

        thrust::fill(d_cellOcc.begin(), d_cellOcc.end(), 0);
        PointHashOperator hashOp(cellOcc);
        thrust::transform(d_pos.begin(), d_pos.end(), d_hash.begin(), hashOp);

        thrust::sort_by_key(
            d_hash.begin(),
            d_hash.end(),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos.begin(),
                                                         d_target.begin()
            )));
        
        thrust::exclusive_scan(d_cellOcc.begin(), d_cellOcc.end(), d_scatter.begin());
        uint maxCellOcc = thrust::reduce(d_cellOcc.begin(), d_cellOcc.end(), 0, thrust::maximum<unsigned int>());
        uint blockSize = 32 * ceil(maxCellOcc / 32.0f);
        dim3 gridSize = dim3(_res, _res);

        for(auto _: state)
        {
            computeAvgNeighbourPos<<<h_gridSize, h_blockSize>>>(collision, targetPos, pos, cellOcc, scatter);
            cudaThreadSynchronize();
        }

    }

} // namespace for benchmark implementation


#define DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_NAME, DT, NUMP, RES)            \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
        benchNeighbour(DT, NUMP, RES, state);                                   \
    }                                                                           \
BENCHMARK(BM_NAME)



DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);