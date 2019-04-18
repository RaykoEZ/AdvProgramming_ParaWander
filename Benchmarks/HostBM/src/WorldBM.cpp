#include "BenchCommon.h"
#include <benchmark/benchmark.h>
#include "World.h"


#define HOST_BM_WORLD_TICK(BM_NAME, NUMP, DT)                                           \
    static void BM_NAME(benchmark::State& state)                                        \
    {                                                                                   \
        float r = 10.0f;                                                                \
        glm::vec3 p = glm::vec3(0.0f);                                                  \
        World testWorld =  World(NUMP,r,p);                                             \
        for(auto _ : state)                                                             \
        {                                                                               \
            benchmark::DoNotOptimize(testWorld.tick(DT));                               \
        }                                                                               \
    }                                                                                   \
BENCHMARK(BM_NAME)

HOST_BM_WORLD_TICK(BM_Host_World_Update_Low, FLOCK_NUMP_LOW, TIMESTEP);
HOST_BM_WORLD_TICK(BM_Host_World_Update_Medium, FLOCK_NUMP_MID, TIMESTEP);
HOST_BM_WORLD_TICK(BM_Host_World_Update_High, FLOCK_NUMP_HIGH, TIMESTEP);
