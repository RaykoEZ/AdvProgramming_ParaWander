#include <benchmark/benchmark.h>
#include "FlockKernels.cuh"
#include "BenchCommon.h"
#include "DeviceTestKernels.cuh"
#include "FlockUtil.cuh"
#include "Hash.cuh"
#include "FlockSystem.h"
#include <vector>


#define DEVICE_BM_BOOL_VECTOR_COPY(BM_NAME, NUMP)                                                   \
    static void BM_NAME(benchmark::State& state)                                                    \
    {                                                                                               \
        std::vector<bool> fH(NUMP);                                                                 \
        for(auto _: state)                                                                          \
        {                                                                                           \
            thrust::device_vector<bool> d_cellOcc(NUMP);                                       \
            d_cellOcc = fH;                                                                         \
        }                                                                                           \
    }                                                                                               \
BENCHMARK(BM_NAME)


#define DEVICE_BM_UINT_VECTOR_COPY(BM_NAME, NUMP)                                                   \
    static void BM_NAME(benchmark::State& state)                                                    \
    {                                                                                               \
        std::vector<uint> fH(NUMP);                                                                 \
        for(auto _: state)                                                                          \
        {                                                                                           \
            thrust::device_vector<uint> d_cellOcc(NUMP);                                       \
            d_cellOcc = fH;                                                                         \
        }                                                                                           \
    }                                                                                               \
BENCHMARK(BM_NAME)


#define DEVICE_BM_FLOAT_VECTOR_COPY(BM_NAME, NUMP)                                                  \
    static void BM_NAME(benchmark::State& state)                                                    \
    {                                                                                               \
        std::vector<float> fH(NUMP);                                                                \
        for(auto _: state)                                                                          \
        {                                                                                           \
            thrust::device_vector<float> d_cellOcc(NUMP);                                       \
            d_cellOcc = fH;                                                                         \
        }                                                                                           \
    }                                                                                               \
BENCHMARK(BM_NAME)

#define DEVICE_BM_FLOAT3_VECTOR_COPY(BM_NAME, NUMP)                                                  \
    static void BM_NAME(benchmark::State& state)                                                     \
    {                                                                                                \
        std::vector<float3> fH(NUMP);                                                                \
        for(auto _: state)                                                                           \
        {                                                                                            \
            thrust::device_vector<float3> d_cellOcc(NUMP);                                       \
            d_cellOcc = fH;                                                                         \
        }                                                                                            \
    }                                                                                                \
BENCHMARK(BM_NAME)



#define DEVICE_BM_INITIALIZE(BM_NAME, DT, NUMP, RES)                                                \
    static void BM_NAME(benchmark::State& state)                                                    \
    {                                                                                               \
        for(auto _: state)                                                                          \
        {                                                                                           \
            FlockSystem flockSys(NUMP,10.0f,0.1f,DT,RES);                                          \
            flockSys.init();                                                                        \
        }                                                                                           \
    }                                                                                               \
BENCHMARK(BM_NAME)


DEVICE_BM_BOOL_VECTOR_COPY(BM_Device_Overhead_Bool_Copy_Single, 1);
DEVICE_BM_BOOL_VECTOR_COPY(BM_Device_Overhead_Bool_Copy_Low, FLOCK_NUMP_LOW);
DEVICE_BM_BOOL_VECTOR_COPY(BM_Device_Overhead_Bool_Copy_Mid, FLOCK_NUMP_MID);
DEVICE_BM_BOOL_VECTOR_COPY(BM_Device_Overhead_Bool_Copy_High, FLOCK_NUMP_HIGH);

DEVICE_BM_UINT_VECTOR_COPY(BM_Device_Overhead_Uint_Copy_Single, 1);
DEVICE_BM_UINT_VECTOR_COPY(BM_Device_Overhead_Uint_Copy_Low, FLOCK_NUMP_LOW);
DEVICE_BM_UINT_VECTOR_COPY(BM_Device_Overhead_Uint_Copy_Mid, FLOCK_NUMP_MID);
DEVICE_BM_UINT_VECTOR_COPY(BM_Device_Overhead_Uint_Copy_High, FLOCK_NUMP_HIGH);

DEVICE_BM_FLOAT_VECTOR_COPY(BM_Device_Overhead_Float_Copy_Single, 1);
DEVICE_BM_FLOAT_VECTOR_COPY(BM_Device_Overhead_Float_Copy_Low, FLOCK_NUMP_LOW);
DEVICE_BM_FLOAT_VECTOR_COPY(BM_Device_Overhead_Float_Copy_Mid, FLOCK_NUMP_MID);
DEVICE_BM_FLOAT_VECTOR_COPY(BM_Device_Overhead_Float_Copy_High, FLOCK_NUMP_HIGH);

DEVICE_BM_FLOAT3_VECTOR_COPY(BM_Device_Overhead_Float3_Copy_Single, 1);
DEVICE_BM_FLOAT3_VECTOR_COPY(BM_Device_Overhead_Float3_Copy_Low, FLOCK_NUMP_LOW);
DEVICE_BM_FLOAT3_VECTOR_COPY(BM_Device_Overhead_Float3_Copy_Mid, FLOCK_NUMP_MID);
DEVICE_BM_FLOAT3_VECTOR_COPY(BM_Device_Overhead_Float3_Copy_High, FLOCK_NUMP_HIGH);

DEVICE_BM_INITIALIZE(BM_Device_Overhead_Initialize_Low, TIMESTEP, FLOCK_NUMP_LOW, FLOCK_RES_LOW);
DEVICE_BM_INITIALIZE(BM_Device_Overhead_Initialize_Mid, TIMESTEP, FLOCK_NUMP_MID, FLOCK_RES_MID);
DEVICE_BM_INITIALIZE(BM_Device_Overhead_Initialize_High, TIMESTEP, FLOCK_NUMP_HIGH, FLOCK_RES_HIGH);
