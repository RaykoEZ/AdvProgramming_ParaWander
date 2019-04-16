#include "FlockParams.cuh"
#include <iostream>

__constant__ FlockData paramData;

FlockParams::FlockParams(const unsigned int &_numB,
                              const float &_m,
                              const float &_vMax,
                              const float &_dt,
                              const unsigned int &_res)
{
    setNumBoids(_numB);
    setMass(_m);
    setInverseMass(_m);
    setVMax(_vMax);
    setTimeStep(_dt);
    setRes(_res);
    
}

void FlockParams::init()
{
    cudaError_t err = cudaMemcpyToSymbol(paramData, &m_data, sizeof(FlockData));
    if (err != cudaSuccess)
    {
        std::cerr << "Copy to symbol params (size=" << sizeof(FlockParams) << ") failed! Reason: " << cudaGetErrorString(err) << "\n";
        exit(0);
    }

}

void FlockParams::setNumBoids(const unsigned int &_numB)
{
    m_data.m_numBoids=_numB;
}

void FlockParams::setMass(const float &_m)
{

    m_data.m_mass=_m;
}

void FlockParams::setInverseMass(const float &_m)
{
    if(_m > 0)
    {
        m_data.m_invMass = 1.0f/_m;
    }
    else return;

}

void FlockParams::setVMax(const float &_vMax)
{
    m_data.m_vMax=_vMax;
}

void FlockParams::setTimeStep(const float &_dt)
{

    m_data.m_dt=_dt;
}

void FlockParams::setRes(const unsigned int &_res)
{
    m_data.m_res = _res;
    m_data.m_res2 = _res * _res;
    m_data.m_invRes = 1.0f/_res;
    m_data.m_invRes2 = m_data.m_invRes * m_data.m_invRes;

}

void FlockParams::setCollisionRad(const float &_rad)
{ 
    m_data.m_collisionRad = _rad;
}