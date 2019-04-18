#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_NEIGHBOURHOOD_SEARCH(BM_NAME)
    static void BM_NAME(benchmark::State& state)\
    {\
        for(auto _: state)\
        {\
            \
        }\
    }\
BENCHMARK(BM_NAME)
