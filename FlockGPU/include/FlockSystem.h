#ifndef FLOCKSYSTEM_H
#define FLOCKSYSTEM_H
#include "FlockDefines.h"
#include <thrust/device_vector.h>

class FlockParams;

class FlockSystem {

public:

    FlockSystem() = delete;
    /// Constructor
    FlockSystem(const uint &_numP,
                const float &_m,
                const float &_vMax,
                const float &_dt,
                const float &_rad);

    /// Destruct our fluid system
    ~FlockSystem();
    void init(const uint &_numP, const uint &_res);
protected:


    void tick(const float &_dt = DEFAULT_TIMESTEP);
    void clear();
    void spawnInRadius(const float &_rad, const float3 &_origin);
    /// Keep track of whether the simulation is ready to start
    bool m_finishedInit;
    float m_spawnRad;
    thrust::device_vector<float3> m_pos;
    thrust::device_vector<float3> m_v;
    thrust::device_vector<float3> m_vMax;
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
