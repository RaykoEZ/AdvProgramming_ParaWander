#ifndef FLOCKPARAMS_CUH
#define FLOCKPARAMS_CUH

#include "FlockDefines.h"
///@brief Struct for storing constants into the GPU
struct FlockData
{
    ///@brief number of boids in the world
    unsigned int m_numBoids;
    ///@brief resolution of the square hash grid
    unsigned int m_res;
    ///@brief resolution squared, usually used for number of grids in the world grid
    unsigned int m_res2;
    ///@brief inversion of res
    float m_invRes;
    ///@brief inversion of res2
    float m_invRes2;
    ///@brief collisionRadius
    float m_collisionRad;
    ///@brief mass of boid
    float m_mass;
    ///@brief inversion of mass
    float m_invMass;
    ///@brief default velocity limit of all boids 
    float m_vMax;
    ///@brief timestep of the world, used for update
    float m_dt;
    
};
///@brief define constant memory prototype here
extern __constant__ FlockData paramData;

/// Class to manage the storage

class FlockParams
{
public:
    ///@brief ctor 
    FlockParams(
    const unsigned int &_numB,
    const float &_m,
    const float &_vMax,
    const float &_dt,
    const unsigned int &_res);
    ///@brief dtor
    ~FlockParams(){}
    ///@brief handle initial memory allocation and boid generation 
    void init();

    ///@brief getters to host from members in device memory
    unsigned int getNumBoids() const { return m_data.m_numBoids;}

    unsigned int getRes() const { return m_data.m_res;}

    unsigned int getRes2() const { return m_data.m_res2;}

    float getInvRes() const { return m_data.m_invRes;}

    float getInvRes2() const { return m_data.m_invRes2;}

    float getMass() const { return m_data.m_mass;}

    float getInverseMass() const { return m_data.m_invMass;}

    float getVMax() const { return m_data.m_vMax;}

    float getTimeStep() const { return m_data.m_dt;}

    float getCollisionRad() const {return m_data.m_collisionRad;}


    ///@brief Setters for members, some are in constant memory
    void setRes(const unsigned int &_res);

    void setNumBoids(const unsigned int &_numB);

    void setMass(const float &_m);

    void setInverseMass(const float &_m);

    void setVMax(const float &_vMax);

    void setTimeStep(const float &_dt);

    void setCollisionRad(const float &_rad);
protected:
    /// @brief data to be sent to constant memory
    FlockData m_data;

};
#endif // FLOCKPARAMS_CUH
