#include <benchmark/benchmark.h>
#include "BenchCommon.h"
#include "Boid.h"
#include "World.h"
#include "glm/vec3.hpp"


#define HOST_BM_BOID_TICK(BM_NAME, DT)\
    static void BM_NAME(benchmark::State& state)\
    {\
        unsigned int n = 1;\
        float r = 10.0f;\
        glm::vec3 p = glm::vec3(0.0f);\
        World testWorld =  World(n,r,p);\
        Boid boid = Boid(0,10.0f,glm::vec3(0.0f),glm::vec3(1.0f,0.0f,0.0f),10.0f, &testWorld);\
        for(auto _ : state)\
        {\
            boid.tick(DT);\
        }\
    }\
BENCHMARK(BM_NAME)



#define HOST_BM_BOID_NEIGHBOUR(BM_NAME, NUMP)                                           \
    static void BM_NAME(benchmark::State& state)                                        \
    {                                                                                   \
        float r = 10.0f;                                                                \
        glm::vec3 p = glm::vec3(0.0f);                                                  \
        World testWorld =  World(NUMP,r,p);                                             \
        for(auto _ : state)                                                             \
        {                                                                               \
            benchmark::DoNotOptimize(testWorld.m_boids[0].getAverageNeighbourPos());    \
        }                                                                               \
    }                                                                                   \
BENCHMARK(BM_NAME)


HOST_BM_BOID_TICK(BM_Host_Boid_Update, TIMESTEP);

HOST_BM_BOID_NEIGHBOUR(BM_Host_Boid_NeighbouthoodProcess_Low, FLOCK_NUMP_LOW);
HOST_BM_BOID_NEIGHBOUR(BM_Host_Boid_NeighbouthoodProcess_Medium, FLOCK_NUMP_MID);
HOST_BM_BOID_NEIGHBOUR(BM_Host_Boid_NeighbouthoodProcess_High, FLOCK_NUMP_HIGH);


