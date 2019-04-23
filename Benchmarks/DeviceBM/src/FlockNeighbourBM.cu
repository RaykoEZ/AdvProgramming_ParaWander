#include <benchmark/benchmark.h>
#include "FlockKernels.cuh"
#include "BenchCommon.h"
#include "DeviceTestKernels.cuh"
#include "FlockUtil.cuh"
#include "Hash.cuh"
#include "FlockSystem.h"
#include <vector>

namespace
{
    static void benchNeighbour(
        const float &_dt,
        const uint &_numP,
        const float &_res, 
        benchmark::State& state)
    {
        for(auto _: state)
        {
            testNeighbour(_dt, _numP, _res);
        }

    }

} // namespace


#define DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_NAME, DT, NUMP, RES)            \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
        benchNeighbour(DT, NUMP, RES, state);                                   \
    }                                                                           \
BENCHMARK(BM_NAME)



DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_Device_Flock_Neighbourhood_Search_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);

