#include <benchmark/benchmark.h>
#include "DeviceTestKernels.cuh"
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

#define DEVICE_BM_FLOCK_HASH(BM_NAME, DT, NUMP, RES)                                                            \
    static void BM_NAME(benchmark::State& state)                                                                \
    {                                                                                                           \
        FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);                                                           \
        flockSys.init();                                                                                        \
        FlockSystem flockSys(_numP,10.0f,0.1f,_dt,_res);                                                        \
        flockSys.init();                                                                                        \
        thrust::device_vector<uint> d_cellOcc = flockSys.getCellOcc();                                          \
        thrust::device_vector<uint> d_scatter = flockSys.getScatterAddress();                                   \
        thrust::device_vector<uint> d_hash = flockSys.getHash();                                                \
        thrust::device_vector<float3> d_pos = flockSys.getPos();                                                \
        uint * cellOcc = thrust::raw_pointer_cast(&d_cellOcc[0]);                                               \                              
        thrust::fill(d_cellOcc.begin(), d_cellOcc.end(), 0);                                                    \
        PointHashOperator hashOp(cellOcc);                                                                      \
        for(auto _: state)                                                                                      \
        {                                                                                                       \
            thrust::transform(d_pos.begin(), d_pos.end(), d_hash.begin(), hashOp);                              \
        }                                                                                                       \
    }                                                                                                           \
BENCHMARK(BM_NAME)



DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_UPDATE_TICK(BM_Device_Flock_Update_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);

DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Flock_Integrator_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Flock_Integrator_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_UPDATE_INTEGRATOR(BM_Device_Flock_Integrator_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);

DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_Medium, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_FLOCK_HASH(BM_Device_Flock_Hash_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);