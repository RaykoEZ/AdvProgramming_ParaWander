#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
#include <vector>


#define DEVICE_BM_FLOCK_BEHAVIOUR_SEEK(BM_NAME)\
    static void BM_NAME(benchmark::State& state)\
    {\
        FlockSystem flockSys(n,mass,0.1f,dt,res);\
        flockSys.init();\
        std::vector<float3> fH {make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> vH {make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> targetH{ make_float3(0.0f,0.0f,0.0f)};\
        float vMax = 10.0f;\
        thrust::device_vector<float3> f = fH;\
        thrust::device_vector<float3> pos = posH;\
        thrust::device_vector<float3> v = vH;\
        thrust::device_vector<float3> target = targetH;\
        for(auto _: state)\
        {\
            testSeek(f, pos, v, target, vMax);\
        }\
    }\
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_FLEE(BM_NAME)\
    static void BM_NAME(benchmark::State& state)\
    {\
        FlockSystem flockSys(n,mass,0.1f,dt,res);\
        flockSys.init();\
        std::vector<float3> fH {make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> vH {make_float3(0.0f,0.0f,0.0f)};\
        std::vector<float3> targetH{ make_float3(0.0f,0.0f,0.0f)};\
        float vMax = 10.0f;\
        thrust::device_vector<float3> f = fH;\
        thrust::device_vector<float3> pos = posH;\
        thrust::device_vector<float3> v = vH;\
        thrust::device_vector<float3> target = targetH;\
        for(auto _: state)\
        {\
            testFlee(f, pos, v, target, vMax);\
        }\
    }\
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOCK_BEHAVIOUR_WANDER(BM_NAME)\
    static void BM_NAME(benchmark::State& state)\
    {\
        FlockSystem flockSys(n,mass,0.1f,dt,res);\
            flockSys.init();\
            std::vector<float> angleH { 0.125f};\
            std::vector<float3> posH { make_float3(0.0f,0.0f,0.0f)};\
            std::vector<float3> vH { make_float3(1.0f,1.0f,0.0f)};\
            std::vector<float3> targetH { make_float3(0.0f,0.0f,0.0f)};\
            thrust::device_vector<float> angle = angleH;\
            thrust::device_vector<float3> pos = posH;\
            thrust::device_vector<float3> v = vH;\
            thrust::device_vector<float3> target = targetH;\
        for(auto _: state)\
        {\
            testWander(target, angle, v, pos);\
        }\
    }\
BENCHMARK(BM_NAME)



DEVICE_BM_FLOCK_BEHAVIOUR_FLEE(BM_Device_Flock_Flee);
DEVICE_BM_FLOCK_BEHAVIOUR_SEEK(BM_Device_Flock_Seek);
DEVICE_BM_FLOCK_BEHAVIOUR_WANDER(BM_Device_Flock_Wander);