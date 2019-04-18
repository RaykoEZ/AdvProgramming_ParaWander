#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_BEHAVIOUR_SEEK(BM_NAME, NUMP)
    static void BM_NAME(benchmark::State& state)\
    {\
        for(auto _: state)\
        {\
            \
        }\
    }\
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_FLEE(BM_NAME, NUMP)
    static void BM_NAME(benchmark::State& state)\
    {\
        for(auto _: state)\
        {\
            \
        }\
    }\
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_WANDER(BM_NAME, NUMP)
    static void BM_NAME(benchmark::State& state)\
    {\
        for(auto _: state)\
        {\
            \
        }\
    }\
BENCHMARK(BM_NAME)



