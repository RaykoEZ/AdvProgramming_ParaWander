#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_UPDATE_TICK(BM_NAME, DT, NUMP, RES)\
    static void BM_NAME(benchmark::State& state)\
    {\
        FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);\
        flockSys.init();\
        for(auto _: state)\
        {\
            flockSys.tick()\
        }\
    }\
BENCHMARK(BM_NAME)



#define DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_NAME, DT, NUMP, RES)\
    static void BM_NAME(benchmark::State& state)\
    {\
        for(auto _: state)\
        {\
            \
        }\
    }\
BENCHMARK(BM_NAME)

