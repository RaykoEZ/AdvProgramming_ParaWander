#include "FlockSystem.h"
#include "Hash.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <random>
#include <vector>
#include <iostream>
#include "FlockParams.cuh"

FlockSystem::FlockSystem(const uint &_numP, const float &_m, const float &_vMax, const float &_dt, const float &_rad)
{
    m_params = new FlockParams(_numP,_m,_vMax,_dt);
    m_params->setNumBoids(_numP);
    if(_m > 0.0f)
    {
        m_params->setMass(_m);
        m_params->setInverseMass(1.0f/_m);

    } 
    else
    {
        m_params->setMass(DEFAULT_MASS);
        m_params->setInverseMass(DEFAULT_MASS_INV);
    }
    m_params->setVMax(_vMax);
    m_params->setTimeStep(_dt);
    m_spawnRad = _rad;
    

}

FlockSystem::~FlockSystem()
{
    clear();
    delete m_params;
}

void FlockSystem::init(const uint &_numP, const uint &_res)
{
    clear();

    m_pos.reserve(m_params->getNumBoids());
    
    spawnInRadius(m_spawnRad,make_float3(0.0f,0.0f,0.0f));
}



void FlockSystem::tick(const float &_dt)
{
    /// We copy the member data over to the gpu
    float3 * pos = thrust::raw_pointer_cast(&m_pos[0]);
    float3 * velocity = thrust::raw_pointer_cast(&m_v[0]);
    float3 * targetPos = thrust::raw_pointer_cast(&m_target[0]);
    float3 * colour = thrust::raw_pointer_cast(&m_col[0]);
    uint * hash = thrust::raw_pointer_cast(&m_hash[0]);
    uint * cellOcc = thrust::raw_pointer_cast(&m_cellOcc[0]);
    uint * scatter = thrust::raw_pointer_cast(&m_scatterAddress[0]);

    thrust::fill(m_cellOcc.begin(), m_cellOcc.end(), 0);

    /// We update boids in gpu below
}

void FlockSystem::clear()
{
    m_pos.clear();
    m_v.clear();
    m_target.clear();
    m_isThereCollision.clear();
    m_hash.clear();
    m_cellOcc.clear();
    m_scatterAddress.clear();


}
void FlockSystem::spawnInRadius(const float &_rad, const float3 &_origin)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());




    std::uniform_real_distribution<float> spawnDis(-_rad, _rad);
    std::uniform_real_distribution<float> vDis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> vMaxDis(1.0f, 10.0f);
}
