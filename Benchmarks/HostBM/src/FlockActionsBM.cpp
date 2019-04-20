#include <benchmark/benchmark.h>
#include "BenchCommon.h"
#include "FlockActions.h"
#include "glm/vec3.hpp"

#define HOST_BM_FLOCK_SEEK(BM_NAME, DT)                                                                 \
    static void BM_NAME(benchmark::State& state)                                                        \
    {                                                                                                   \  
        glm::vec3 pos(0.0f);                                                                            \
        glm::vec3 v(1.0f, 0.0f, 0.0f);                                                                  \
        float vMax = 10.0f;                                                                             \
        glm::vec3 target(10.0f,0.0f,0.0f);                                                              \
        for(auto _ : state)                                                                             \
        {                                                                                               \
            benchmark::DoNotOptimize(seek(pos, v, vMax, target));                                       \
        }                                                                                               \
    }                                                                                                   \
BENCHMARK(BM_NAME)


#define HOST_BM_FLOCK_FLEE(BM_NAME, DT)                                                                 \
    static void BM_NAME(benchmark::State& state)                                                        \
    {                                                                                                   \
        glm::vec3 pos(0.0f);                                                                            \
        glm::vec3 v(1.0f, 0.0f, 0.0f);                                                                  \
        float vMax = 10.0f;                                                                             \
        glm::vec3 target(10.0f,0.0f,0.0f);                                                              \
        for(auto _ : state)                                                                             \
        {                                                                                               \
            benchmark::DoNotOptimize(flee(pos, v, vMax, target));                                       \
        }                                                                                               \
    }                                                                                                   \
BENCHMARK(BM_NAME)


#define HOST_BM_FLOCK_WANDER(BM_NAME, DT)                                                               \
    static void BM_NAME(benchmark::State& state)                                                        \
    {                                                                                                   \
        glm::vec3 pos(0.0f);                                                                            \
        glm::vec3 v(1.0f, 0.0f, 0.0f);                                                                  \
        for(auto _ : state)                                                                             \
        {                                                                                               \
            benchmark::DoNotOptimize(wander(pos, v));                                                   \
        }                                                                                               \
    }                                                                                                   \
BENCHMARK(BM_NAME)