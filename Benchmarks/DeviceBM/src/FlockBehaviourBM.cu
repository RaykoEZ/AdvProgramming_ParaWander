#include <benchmark/benchmark.h>
#include "BenchCommon.h"
#include "DeviceTestKernels.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_BEHAVIOUR_SEEK(BM_NAME, DT)                             \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
        FlockSystem flockSys(1,10.0F,0.1f,DT,128.0f);                           \
        flockSys.init();                                                        \
        std::vector<float3> fH {make_float3(0.0f,0.0f,0.0f)};                   \
        std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};                \
        std::vector<float3> vH {make_float3(0.0f,0.0f,0.0f)};                   \
        std::vector<float3> targetH{ make_float3(0.0f,0.0f,0.0f)};              \
        float vMax = 10.0f;                                                     \
        thrust::device_vector<float3> f = fH;                                   \
        thrust::device_vector<float3> pos = posH;                               \
        thrust::device_vector<float3> v = vH;                                   \
        thrust::device_vector<float3> target = targetH;                         \
        for(auto _: state)                                                      \
        {                                                                       \
            testSeek(f, pos, v, target, vMax);                                  \
        }                                                                       \
    }                                                                           \
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_FLEE(BM_NAME, DT)                             \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
        FlockSystem flockSys(1,10.0F,0.1f,DT,128.0f);                           \
        flockSys.init();                                                        \
        std::vector<float3> fH {make_float3(0.0f,0.0f,0.0f)};                   \
        std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};                \
        std::vector<float3> vH {make_float3(0.0f,0.0f,0.0f)};                   \
        std::vector<float3> targetH{ make_float3(0.0f,0.0f,0.0f)};              \
        float vMax = 10.0f;                                                     \
        thrust::device_vector<float3> f = fH;                                   \
        thrust::device_vector<float3> pos = posH;                               \
        thrust::device_vector<float3> v = vH;                                   \
        thrust::device_vector<float3> target = targetH;                         \
        for(auto _: state)                                                      \
        {                                                                       \
            testFlee(f, pos, v, target, vMax);                                  \
        }                                                                       \
    }                                                                           \
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_WANDER(BM_NAME, DT)                           \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
            FlockSystem flockSys(1,10.0F,0.1f,DT,128.0f);                       \
            flockSys.init();                                                    \
            std::vector<float> angleH { 0.125f};                                \
            std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};            \
            std::vector<float3> vH { make_float3(1.0f,1.0f,0.0f)};              \
            std::vector<float3> targetH { make_float3(0.0f,0.0f,0.0f)};         \
            thrust::device_vector<float> angle = angleH;                        \
            thrust::device_vector<float3> pos = posH;                           \
            thrust::device_vector<float3> v = vH;                               \
            thrust::device_vector<float3> target = targetH;                     \
        for(auto _: state)                                                      \
        {                                                                       \
            testWander(target, angle, v, pos);                                  \
        }                                                                       \
    }                                                                           \
BENCHMARK(BM_NAME)

#define DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_NAME, DT, NUMP, RES)               \
    static void BM_NAME(benchmark::State& state)                                \
    {                                                                           \
        FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);                           \
        flockSys.init();                                                        \
        std::vector<float3> posH(NUMP);                                         \
        std::vector<float3> vH(NUMP);                                           \
        std::vector<float3> fH(NUMP);                                           \
        for(uint i = 0; i< NUMP; ++i)                                           \
        {                                                                       \
            posH.push_back(make_float3(0.0f,0.0f,0.0f));                        \
            vH.push_back(make_float3(1.0f,0.0f,0.0f));                          \
            fH.push_back(make_float3(0.0f,0.0f,0.0f));                          \
        }                                                                       \
        thrust::device_vector<float3> pos = posH;                               \
        thrust::device_vector<float3> v = vH;                                   \
        thrust::device_vector<float3> f = fH;                                   \
        float vMax = 10.0f;                                                     \
        for(auto _: state)                                                      \
        {                                                                       \
            testResolveForce(pos, v, f, vMax);                                  \
        }                                                                       \
    }                                                                           \
BENCHMARK(BM_NAME)

DEVICE_BM_FLOCK_BEHAVIOUR_FLEE(BM_Device_Flock_Flee, TIMESTEP);
DEVICE_BM_FLOCK_BEHAVIOUR_SEEK(BM_Device_Flock_Seek, TIMESTEP);
DEVICE_BM_FLOCK_BEHAVIOUR_WANDER(BM_Device_Flock_Wander, TIMESTEP);
DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Flock_Integrator, TIMESTEP, 1, FLOCK_RES_HIGH);
