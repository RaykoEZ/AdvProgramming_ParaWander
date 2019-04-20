#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
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



#define DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_NAME, DT, NUMP, RES)   \
    static void BM_NAME(benchmark::State& state)                    \
    {                                                               \
        FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);               \
        flockSys.init();                                            \
        std::vector<float3> posH(NUMP);                             \
        std::vector<float3> vH(NUMP);                               \
        std::vector<float3> fH(NUMP);                               \
        for(uint i = 0; i< NUMP; ++i)                               \
        {                                                           \
            posH.push_back(make_float3(0.0f,0.0f,0.0f));            \
            vH.push_back(make_float3(1.0f,0.0f,0.0f));              \
            fH.push_back(make_float3(0.0f,0.0f,0.0f));              \
        }                                                           \
        thrust::device_vector<float3> pos = posH;                   \
        thrust::device_vector<float3> v = vH;                       \
        thrust::device_vector<float3> f = fH;                       \   
        float vMax = 10.0f;                                         \
        for(auto _: state)                                          \
        {                                                           \
            testResolveForce(pos, v, f, vMax);                      \
        }                                                           \
    }                                                               \
BENCHMARK(BM_NAME)

DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);

DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Integrator_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Integrator_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Integrator_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);