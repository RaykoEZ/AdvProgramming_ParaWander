#include "FlockSystem.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <thrust/device_vector.h>

#include <random>
#include <time.h>
#include <vector>
#include <iostream>

#include "FlockParams.cuh"

FlockSystem::FlockSystem(const unsigned int &_numP, const float &_m, const float &_vMax, const float &_dt)
{
    m_params = new FlockParams(_numP,_m,_vMax,_dt);

}

FlockSystem::~FlockSystem()
{
    clear();
    delete m_params;
}

void FlockSystem::init(const unsigned int &_numP, const unsigned int &_res)
{
    clear();

}


void FlockSystem::setup(const unsigned int &_numP, const unsigned int &_res)
{



}

void FlockSystem::tick(const float &_dt)
{

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
void FlockSystem::createSpawnCircle(const float &_rad, const float3 &_origin)
{

}
