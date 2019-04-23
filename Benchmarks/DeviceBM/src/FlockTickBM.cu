#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
#include "BenchCommon.h"
#include "FlockKernels.cuh"
#include "Hash.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_UPDATE_TICK(BM_NAME, DT, NUMP, RES)         \
    static void BM_NAME(benchmark::State& state)                    \
    {                                                               \
        FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);               \
        flockSys.init();                                            \
        for(auto _: state)                                          \
        {                                                           \
            flockSys.tick();                                        \
        }                                                           \
    }                                                               \
BENCHMARK(BM_NAME)




#define DEVICE_BM_FLOCK_HASH(BM_NAME, DT, NUMP, RES)                                                            \
    static void BM_NAME(benchmark::State& state)                                                                \
    {                                                                                                           \
        for(auto _: state)                                                                                      \
        {                                                                                                       \
           testHash(DT, NUMP, RES);                                                                             \
        }                                                                                                       \
    }                                                                                                           \
BENCHMARK(BM_NAME)



DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);


DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);


