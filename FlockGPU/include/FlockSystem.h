#ifndef FLOCKSYSTEM_H
#define FLOCKSYSTEM_H
#include "FlockDefines.h"
#include <thrust/device_vector.h>

class FlockParams;

class FlockSystem {

public:
    /// Constructor
    FlockSystem(const uint &_numP,
                const float &_m,
                const float &_vMax,
                const float &_dt);

    /// Destruct our fluid system
    ~FlockSystem();

    /// Initialise a relatively standard dambreak simulation
    void setup(const uint &_numP, const uint &_res);
protected:
    /// Keep track of whether the simulation is ready to start
    bool m_finishedInit;

    void init(const uint &_numP, const uint &_res);

    void tick(const float &_dt = DEFAULT_TIMESTEP);
    void clear();
    void spawnInRadius(const float &_rad, const float3 &_origin);

    thrust::device_vector<float3> m_pos;
    thrust::device_vector<float3> m_v;
    thrust::device_vector<float3> m_target;
    thrust::device_vector<float3> m_col;
    thrust::device_vector<uint> m_hash;
    thrust::device_vector<uint> m_cellOcc;
    thrust::device_vector<uint> m_scatterAddress;
    thrust::device_vector<bool> m_isThereCollision;






private:
    FlockParams* m_params;

};

#endif //FLOCK_SYSTEM_H
