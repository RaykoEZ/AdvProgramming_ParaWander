#include "FlockSystem.h"
#include "Hash.cuh"
#include "DebugUtil.cuh"
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
    h_frameCount = 0;
    h_init = false;
    

}

FlockSystem::~FlockSystem()
{
    clear();
    h_frameCount = 0;
    delete h_params;
}

void FlockSystem::init()
{
    clear();
    //h_params->init();
    /// resize vectors for future storage
    d_pos.resize(h_params->getNumBoids());
    d_v.resize(h_params->getNumBoids());
    d_vMax.resize(h_params->getNumBoids());
    d_target.resize(h_params->getNumBoids());
    d_col.resize(h_params->getNumBoids());

    d_angle.resize(h_params->getNumBoids());
    d_hash.resize(h_params->getNumBoids(),0);
    d_cellOcc.resize(h_params->getRes2(),0);
    d_scatterAddress.resize(h_params->getRes2());
    d_isThereCollision.resize(h_params->getNumBoids());

    thrust::fill(d_col.begin(), d_col.end(),make_float3(0.0f,255.0f,0.0f));
    thrust::fill(d_isThereCollision.begin(), d_isThereCollision.end(),false);

    
    prepareBoids(h_params->getNumBoids(), 0.1f,0.1f,
                                          0.9f,0.9f);




    //cudaErrorPrint();
    h_params->init();
}



void FlockSystem::tick()
{
    if(!h_init) return;




    /// We cast to raw ptr for kernel calls
    float3 * pos = thrust::raw_pointer_cast(&d_pos[0]);
    float3 * velocity = thrust::raw_pointer_cast(&d_v[0]);
    float3 * targetPos = thrust::raw_pointer_cast(&d_target[0]);
    float3 * colour = thrust::raw_pointer_cast(&d_col[0]);
    float * vMax = thrust::raw_pointer_cast(&d_vMax[0]);
    bool * collision = thrust::raw_pointer_cast(&d_isThereCollision[0]);
    float * angle = thrust::raw_pointer_cast(&d_angle[0]);
    uint * cellOcc = thrust::raw_pointer_cast(&d_cellOcc[0]);
    uint * scatter = thrust::raw_pointer_cast(&d_scatterAddress[0]);

    /// Set random floats for boid wandering search angle
    randomFloats(angle, h_params->getNumBoids(),h_frameCount);
    /// flush prior occupancy out and put new occupancy data in
    thrust::fill(d_cellOcc.begin(), d_cellOcc.end(), 0);
    PointHashOperator hashOp(cellOcc);
    thrust::transform(d_pos.begin(), d_pos.end(), d_hash.begin(), hashOp);




    //cudaErrorPrint();

    thrust::sort_by_key(
    d_hash.begin(),
    d_hash.end(),
    thrust::make_zip_iterator(thrust::make_tuple(d_pos.begin(),
                                                 d_v.begin(),
                                                 d_target.begin(),
                                                 d_angle.begin(),
                                                 d_vMax.begin()
    )));
                    



    //cudaErrorPrint();
    thrust::exclusive_scan(d_cellOcc.begin(), d_cellOcc.end(), d_scatterAddress.begin());
    //cudaErrorPrint();
    uint maxCellOcc = thrust::reduce(d_cellOcc.begin(), d_cellOcc.end(), 0, thrust::maximum<unsigned int>());

    /// define block dims to solve for ths frame
    uint blockSize = 32 * ceil(maxCellOcc / 32.0f);
    dim3 gridSize(h_params->getRes(), h_params->getRes());
    //std::cout<<"Res dump: "<< h_params->getRes()<<'\n';
    //d_threadIdxCheck.resize(blockSize);
    //d_blockIdxCheck.resize(h_params->getRes2());

    //uint * thread = thrust::raw_pointer_cast(&d_threadIdxCheck[0]);
    //uint * block = thrust::raw_pointer_cast(&d_blockIdxCheck[0]);
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

    //cudaErrorPrint();
    /// now we decide to wander if no collision, flee if there is collision
    genericBehaviour<<<gridSize,blockSize>>>(
                                               velocity,
                                               colour,
                                               targetPos,
                                               pos,
                                               collision,
                                               cellOcc,
                                               scatter,
                                               angle,
                                               vMax);
    cudaThreadSynchronize();

    //cudaErrorPrint();
    ++h_frameCount;
}

void FlockSystem::clear()
{
    d_pos.clear();
    d_v.clear();
    d_vMax.clear();
    d_target.clear();
    d_col.clear();
    d_angle.clear();
    d_isThereCollision.clear();
    d_hash.clear();
    d_cellOcc.clear();
    d_scatterAddress.clear();
    
    //d_threadIdxCheck.clear();
    //d_blockIdxCheck.clear();


}



void FlockSystem::prepareBoids(const float &_nBoids,
                               const float &_minX, const float &_minY,
                               const float &_maxX, const float &_maxY)
{

    float3 minCorner = make_float3(_minX, _minY, 0.0f);
    float3 maxCorner = make_float3(_maxX, _maxY, 0.0f);

    float3 diff = maxCorner - minCorner;
    float3 halfDiff = 0.5f * diff;
    float3 mid = make_float3(minCorner.x + halfDiff.x,minCorner.y + halfDiff.y,0.0f);


    std::random_device rd;
    std::mt19937_64 gen(rd());

    float3 quartDiff = 0.5f* halfDiff;
    float rad = length(quartDiff);
    float boidRad =0.5f * h_params->getInvRes();
    h_params->setCollisionRad(boidRad);
    std::uniform_real_distribution<float> spawnDis(0.1f, 0.9f);

    std::uniform_real_distribution<float> vDis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> vMaxDis(1.0f, 10.0f);



    std::vector<float3> posHost;
    std::vector<float3> vHost;
    std::vector<float> vMaxHost;

    float3 pos;
    float3 v;
    for(unsigned int i = 0; i < _nBoids; ++i)
    {
        pos = make_float3(spawnDis(gen),spawnDis(gen), 0.0f);

        //std::cout<< pos.x<<", "<< pos.y<< ", "<< pos.z<<'\n';
        v = make_float3(vDis(gen), vDis(gen), 0.0f);
        posHost.push_back(pos);
        vHost.push_back(v);
        vMaxHost.push_back(vMaxDis(gen));
    }
    /// copy pos and velocity results to device vector
    thrust::copy(posHost.begin(),posHost.end(),d_pos.begin());
    thrust::copy(vHost.begin(),vHost.end(),d_v.begin());
    thrust::copy(vMaxHost.begin(),vMaxHost.end(),d_vMax.begin());

    h_init = true;


}
void FlockSystem::exportResult(std::vector<float3> &_posh, std::vector<float3> &_colh) const
{
    thrust::copy(d_col.begin(), d_col.end(), _colh.begin());
    thrust::copy(d_pos.begin(), d_pos.end(), _posh.begin());

    
    //std::vector<uint> hashH;
    //hashH.resize(h_params->getNumBoids());
    //thrust::copy(d_hash.begin(),d_hash.end(), hashH.begin());
    //std::vector<float3> vh;
    //vh.resize(h_params->getNumBoids());
    //std::vector<float3> targeth;
    //targeth.resize(h_params->getNumBoids());
    //thrust::copy(d_v.begin(), d_v.end(),vh.begin());
    //thrust::copy(d_target.begin(), d_target.end(),targeth.begin());

    //std::vector<bool> collisionh;
    //std::vector<float> angleh;
    //std::vector<uint> occh;
    //occh.resize(h_params->getRes2());
    //angleh.resize(h_params->getNumBoids());
    //collisionh.resize(h_params->getNumBoids());
    //thrust::copy(d_cellOcc.begin(), d_cellOcc.end(),occh.begin());
    //thrust::copy(d_isThereCollision.begin(), d_isThereCollision.end(),collisionh.begin());
    //thrust::copy(d_angle.begin(), d_angle.end(),angleh.begin());
    //std::cout<<"Num of Cells allocated: "<< h_params->getRes2() << '\n';
    //uint size = h_params->getNumBoids();
    
    //for(int i = 0; i< size;++i)
    //{
        //std::cout<< "Collision check: "<< collisionh[i] << '\n';
        //std::cout<< "Hash check: "<< hashH[i] << '\n';

        //std::cout<<"Angle Check: " << angleh[i] * 360.0f <<'\n';
        //std::cout<<"Pos Check: "<< _posh[i].x << ", " << _posh[i].y<< '\n';
        //std::cout<<"V Check: "<< vh[i].x << ", " << vh[i].y<< '\n';
        //std::cout<<"Target Check: "<< targeth[i].x << ", " << targeth[i].y<< '\n';

    //}
    
    //std::vector<uint> threadH;
    //std::vector<uint> blockH;
    //threadH.resize(d_threadIdxCheck.size());
    //blockH.resize(d_blockIdxCheck.size());

    //thrust::copy(d_threadIdxCheck.begin(),d_threadIdxCheck.end(),threadH.begin());
    //thrust::copy(d_blockIdxCheck.begin(),d_blockIdxCheck.end(),blockH.begin());

    //for(int i = 0; i< threadH.size();++i)
    //{
        
    //    std::cout<< "ThreadIdx check: "<< threadH[i]<<'\n';

        //std::cout<< "Occupancy check: "<< occh[i] << '\n'; 
    //}
    /*
    for(int i = 0; i< h_params->getRes2();++i)
    {
        
        //std::cout<< "BlockIdx check: "<< blockH[i].x << ", "<< blockH[i].y<<", "<<blockH[i].z<<'\n';

        std::cout<< "Occupancy check in blocks: "<< blockH[i] << '\n'; 
    }*/
    



}
