#include "FlockSystem.h"
#include "Hash.cuh"
#include <random>
#include <vector>
#include <iostream>
#include "FlockParams.cuh"
#include "FloackKernels.cuh"
#include "Random.cuh"

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
    /// reserve vectors for future storage
    m_pos.reserve(m_params->getNumBoids());
    m_v.reserve(m_params->getNumBoids());
    m_target.reserve(m_params->getNumBoids());
    m_col.reserve(m_params->getNumBoids());
    m_angle.reserve(m_params->getNumBoids());    
    m_hash.reserve(m_params->getNumBoids());
    m_cellOcc.reserve(m_params->getNumBoids());
    m_scatterAddress.reserve(m_params->getNumBoids());
    m_isThereCollision.reserve(m_params->getNumBoids());

    
    prepareBoids(m_spawnRad,make_float3(0.0f,0.0f,0.0f));
}



void FlockSystem::tick(const float &_dt)
{
    /// We cast to raw ptr for kernel calls
    float3 * pos = thrust::raw_pointer_cast(&m_pos[0]);
    float3 * velocity = thrust::raw_pointer_cast(&m_v[0]);
    float3 * targetPos = thrust::raw_pointer_cast(&m_target[0]);
    
    bool * collision = thrust::raw_pointer_cast(&m_isThereCollision[0]);
    float * angle = thrust::raw_pointer_cast(&m_angle[0]);
    uint * hash = thrust::raw_pointer_cast(&m_hash[0]);
    uint * cellOcc = thrust::raw_pointer_cast(&m_cellOcc[0]);
    uint * scatter = thrust::raw_pointer_cast(&m_scatterAddress[0]);
    /// copy global parameters to gpu
    m_params->init()

    /// flush prior occupancy out and put new occupancy data in
    thrust::fill(m_cellOcc.begin(), m_cellOcc.end(), 0);
    PointHashOperator hashOp(cellOcc);
    thrust::transform(m_pos.begin(), m_pos.end(), m_hash.begin(), hashOp);
    
    thrust::sort_by_key(m_Hash.begin(),
    m_Hash.end(),
    thrust::make_zip_iterator(thrust::make_tuple(m_pos.begin(),m_v.begin(), m_target.begin(), m_isThereCollision.begin()));
   
    thrust::exclusive_scan(m_cellOcc.begin(), m_cellOcc.end(), m_scatterAddress.begin());
    uint maxCellOcc = thrust::reduce(m_cellOcc.begin(), m_cellOcc.end(), 0, thrust::maximum<unsigned int>());
   
    /// define block dims to solve for ths frame
    uint blockSize = 32 * ceil(maxCellOcc / 32.0f);
    dim3 gridSize(m_params->getRes(), m_params->getRes());
    /// We update boids in gpu below

    /// Set random floats for boid wandering search angle
    randomFloats(angle, m_params->getNumBoids());

    /// Spatial hash values, cell occupancy, memory scatter offset ( scatter addresses), positions and direction are already initialized, now we:
    /// - determine neighbourhood and collision flag
    /// - calculate target position and behaviour depending on collision flag
    /// - resolve forces
    /// - change colours if colliding

    /// Modifies: 
    /// - collision flag
    /// - Boid Target Position (to average neighbourhood position)
    computeAvgNeighbourPos<<<gridSize, blockSize>>>(collision, targetPos, pos, cellOcc, scatter);
    cudaThreadSynchronize();

    /// now we decide to wander if no collision, flee if there is collision




}

void FlockSystem::clear()
{
    m_pos.clear();
    m_v.clear();
    m_target.clear();
    m_col.clear();
    m_angle.clear();
    m_isThereCollision.clear();
    m_hash.clear();
    m_cellOcc.clear();
    m_scatterAddress.clear();


}
void FlockSystem::prepareBoids(const float &_rad, const float3 &_origin)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());

    std::uniform_real_distribution<float> spawnDis(-_rad, _rad);
    std::uniform_real_distribution<float> vDis(-1.0f, 1.0f);
    //std::uniform_real_distribution<float> vMaxDis(1.0f, 10.0f);

    thrust::host_vector posHost;
    thrust::host_vector vHost;
    float3 pos;
    float3 v;
    for(unsigned int i = 0; i < _nBoids; ++i)
    {
        pos = _origin + make_float3(spawnDis(gen),spawnDis(gen), 0.0f);
        v = make_float3(vDis(gen), vDis(gen), 0.0f);
        posHost.push_back(pos);
        vHost.push_back(v);
    }
    /// copy pos and velocity results to device vector
    m_pos = posHost;
    m_v = vHost;

}
