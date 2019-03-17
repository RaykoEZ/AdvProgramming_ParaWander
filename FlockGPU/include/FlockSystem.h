#ifndef FLOCKSYSTEM_H
#define FLOCKSYSTEM_H
#include "FlockDefines.h"
#include <vector>

class FlockParams;

class FlockSystem {

public:
    /// Constructor
    FlockSystem(const unsigned int &_numP,
                const float &_m,
                const float &_vMax,
                const float &_dt);

    /// Destruct our fluid system
    ~FlockSystem();

    /// Initialise a relatively standard dambreak simulation
    void setup(const unsigned int &_numP, const unsigned int &_res);
protected:
    /// Keep track of whether the simulation is ready to start
    bool m_finishedInit;

    void init(const unsigned int &_numP, const unsigned int &_res);

    void tick(const float &_dt = DEFAULT_TIMESTEP);
    void clear();
    void createSpawnCircle(const float &_rad, const float3 &_origin);

    thrust::vector<float3> m_pos;
    thrust::vector<float3> m_v;
    thrust::vector<float3> m_f;
    thrust::vector<float3> m_col;


    thrust::vector<uint> m_hash;
    thrust::vector<uint> m_cellOcc;;






private:
    FlockParams* m_params;

};

#endif //FLOCK_SYSTEM_H
