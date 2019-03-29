#include "FlockSystem.h"
#include "Hash.cuh"
#include <random>

#include <iostream>
#include "FlockParams.cuh"
#include "FlockKernels.cuh"
#include "Random.cuh"


FlockSystem::FlockSystem(const uint &_numP, const float &_m, const float &_vMax, const float &_dt, const float &_rad, const float &_res)
{
    h_params = new FlockParams(_numP,_m,_vMax,_dt,_res);
    h_params->setNumBoids(_numP);
    if(_m > 0.0f)
    {
        h_params->setMass(_m);
        h_params->setInverseMass(1.0f/_m);

    } 
    else
    {
        h_params->setMass(DEFAULT_MASS);
        h_params->setInverseMass(DEFAULT_MASS_INV);
    }
    h_params->setVMax(_vMax);
    h_params->setTimeStep(_dt);
    h_spawnRad = _rad;
    h_init = false;
    

}

FlockSystem::~FlockSystem()
{
    clear();
    delete h_params;
}

void FlockSystem::init()
{
    clear();
    h_params->init();
    /// reserve vectors for future storage
    d_pos.resize(h_params->getNumBoids());
    d_v.resize(h_params->getNumBoids());
    d_target.resize(h_params->getNumBoids());
    d_col.resize(h_params->getNumBoids());
    d_angle.resize(h_params->getNumBoids());
    d_hash.resize(h_params->getNumBoids(),0);
    d_cellOcc.resize(h_params->getRes2(),0);
    d_scatterAddress.resize(h_params->getRes2());
    d_isThereCollision.resize(h_params->getNumBoids());

    
    prepareBoids(h_params->getNumBoids(), 0.1f,0.1f,
                                          0.9f,0.9f);


    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Thrust allocation failed, error " << cudaGetErrorString(err) << "\n";
            exit(0);
    }

    h_init = true;
}



void FlockSystem::tick()
{
    if(!h_init) return;

    /// We cast to raw ptr for kernel calls
    float3 * pos = thrust::raw_pointer_cast(&d_pos[0]);
    float3 * velocity = thrust::raw_pointer_cast(&d_v[0]);
    float3 * targetPos = thrust::raw_pointer_cast(&d_target[0]);
    float3 * colour = thrust::raw_pointer_cast(&d_col[0]);
    bool * collision = thrust::raw_pointer_cast(&d_isThereCollision[0]);
    float * angle = thrust::raw_pointer_cast(&d_angle[0]);
    uint * cellOcc = thrust::raw_pointer_cast(&d_cellOcc[0]);
    uint * scatter = thrust::raw_pointer_cast(&d_scatterAddress[0]);

    /// Set random floats for boid wandering search angle
    randomFloats(angle, h_params->getNumBoids());
    /// flush prior occupancy out and put new occupancy data in
    thrust::fill(d_cellOcc.begin(), d_cellOcc.end(), 0);
    PointHashOperator hashOp(cellOcc);
    thrust::transform(d_pos.begin(), d_pos.end(), d_hash.begin(), hashOp);




    thrust::sort_by_key(
    d_hash.begin(),
    d_hash.end(),
    thrust::make_zip_iterator(thrust::make_tuple(d_pos.begin(),
                                                 d_v.begin(),
                                                 d_target.begin(),
                                                 d_angle.begin(),
                                                 d_isThereCollision.begin()
                                                 )));

    thrust::exclusive_scan(d_cellOcc.begin(), d_cellOcc.end(), d_scatterAddress.begin());
    uint maxCellOcc = thrust::reduce(d_cellOcc.begin(), d_cellOcc.end(), 0, thrust::maximum<unsigned int>());

    /// define block dims to solve for ths frame
    uint blockSize = 32 * ceil(maxCellOcc / 32.0f);
    dim3 gridSize(h_params->getRes(), h_params->getRes());
    /// We update boids in gpu below

    /// Spatial hash values, cell occupancy, memory scatter offset ( scatter addresses),
    /// positions and direction are already initialized, now we:
    /// - determine neighbourhood and collision flag
    /// - calculate target position and behaviour depending on collision flag
    /// - resolve forces
    /// - change colours if colliding

    /// Modifies:
    /// - collision flag
    /// - Boid Target Position (to average neighbourhood position)
    ///

    std::cout << "maxCellOcc=" << maxCellOcc << ", blockSize=" << blockSize << ", gridSize=" << h_params->getRes() << "^2\n";

    computeAvgNeighbourPos<<<gridSize, blockSize>>>(collision, targetPos, pos, cellOcc, scatter);
    cudaThreadSynchronize();
    /// now we decide to wander if no collision, flee if there is collision
    genericBehaviour<<<gridSize,blockSize>>>(
                                               velocity,
                                               colour,
                                               targetPos,
                                               pos,
                                               collision,
                                               cellOcc,
                                               scatter,
                                               angle);
    cudaThreadSynchronize();




}

void FlockSystem::clear()
{
    d_pos.clear();
    d_v.clear();
    d_target.clear();
    d_col.clear();
    d_angle.clear();
    d_isThereCollision.clear();
    d_hash.clear();
    d_cellOcc.clear();
    d_scatterAddress.clear();


}



void FlockSystem::prepareBoids(const float &_nBoids,
                               const float &_minX, const float &_minY,
                               const float &_maxX, const float &_maxY)
{

    float3 minCorner = make_float3(_minX, _minY, 0.0f);
    float3 maxCorner = make_float3(_maxX, _maxY, 0.0f);

    float3 diff = maxCorner - minCorner;
    float3 halfDiff = 0.5f * diff;
    float3 mid = minCorner + halfDiff;


    std::random_device rd;
    std::mt19937_64 gen(rd());

    float rad = length(diff);
    std::uniform_real_distribution<float> spawnDis(-rad, rad);

    std::uniform_real_distribution<float> vDis(-1.0f, 1.0f);
    //std::uniform_real_distribution<float> vMaxDis(1.0f, 10.0f);

    std::vector<float3> posHost;
    std::vector<float3> vHost;
    float3 pos;
    float3 v;
    for(unsigned int i = 0; i < _nBoids; ++i)
    {
        pos = mid + make_float3(spawnDis(gen),spawnDis(gen), 0.0f);

        std::cout<< pos.x<<", "<< pos.y<< ", "<< pos.z<<'\n';
        v = make_float3(vDis(gen), vDis(gen), 0.0f);
        posHost.push_back(pos);
        vHost.push_back(v);
    }
    /// copy pos and velocity results to device vector
    thrust::copy(posHost.begin(),posHost.end(),d_pos.begin());
    thrust::copy(vHost.begin(),vHost.end(),d_v.begin());



}
void FlockSystem::exportResult(std::vector<float3> &_posh, std::vector<float3> &_colh) const
{
    //std::cout<< "finished" << '\n';
    thrust::copy(d_col.begin(), d_col.end(), _colh.begin());
    thrust::copy(d_pos.begin(), d_pos.end(), _posh.begin());

}
