#ifndef FLOCKPARAMS_CUH
#define FLOCKPARAMS_CUH

#include "FlockDefines.h"
/// Struct for storing constants into the GPU

struct FlockData
{
    unsigned int m_numBoids;
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
    const float &_dt);

    ~FlockParams();

    void init();

    unsigned int getNumBoids() const { return m_data.m_numBoids;}

    float getMass() const { return m_data.m_mass;}

    float getInverseMass() const { return m_data.m_invMass;}

    float getVMax() const { return m_data.m_vMax;}

    float getTimeStep() const { return m_data.m_dt;}




    void setNumBoids(const unsigned int &_numB);

    void setMass(const float &_m);

    void setInverseMass(const float &_m);

    void setVMax(const float &_vMax);

    void setTimeStep(const float &_dt);

protected:

    FlockData m_data;

};
#endif // FLOCKPARAMS_CUH
