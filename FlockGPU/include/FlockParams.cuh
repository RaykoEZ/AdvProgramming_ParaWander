#ifndef FLOCKPARAMS_CUH
#define FLOCKPARAMS_CUH

#include "FlockDefines.h"
/// Struct for storing constants into the GPU

struct FlockData
{
    unsigned int m_numBoids;
    unsigned int m_res;
    unsigned int m_res2;
    float m_invRes;
    float m_invRes2;
    float m_collisionRad;

    float m_mass;
    float m_invMass;
    float m_vMax;
    float m_dt;
    
};

extern __constant__ FlockData paramData;

/// Class to manage the storage

class FlockParams
{
public:
    FlockParams(
    const unsigned int &_numB,
    const float &_m,
    const float &_vMax,
    const float &_dt,
    const unsigned int &_res);

    ~FlockParams(){}

    void init();

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


    /// Recalculates res2 and their inversions
    void setRes(const unsigned int &_res);

    void setNumBoids(const unsigned int &_numB);

    void setMass(const float &_m);

    void setInverseMass(const float &_m);

    void setVMax(const float &_vMax);

    void setTimeStep(const float &_dt);

    void setCollisionRad(const float &_rad);
protected:

    FlockData m_data;

};
#endif // FLOCKPARAMS_CUH
