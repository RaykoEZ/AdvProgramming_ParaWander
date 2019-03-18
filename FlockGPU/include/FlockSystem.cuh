#ifndef FLOCKSYSTEM_CUH
#define FLOCKSYSTEM_CUH
#include "FlockDefines.h"
#include <vector>
#include <thrust/device_vector.h>

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


    thrust::device_vector<float3> m_pos;
    thrust::device_vector<float3> m_v;
    thrust::device_vector<float3> m_target;
    thrust::device_vector<bool> m_isThereCollision;

    /// Individual point hash for each point - length numPoints
    thrust::device_vector<uint> m_hash;

    /// Cell occupancy count for each cell - length numCells = res^2
    thrust::device_vector<uint> m_cellOcc;

    /// Store the scatter addresses to find the start position of all the cells in GPU memory. Size numCells
    thrust::device_vector<uint> m_scatterAddress;



    void init(const unsigned int &_numP, const unsigned int &_res);

    void tick(const float &_dt = DEFAULT_TIMESTEP);
    void clear();
private:
    FlockParams* m_params;

};
#endif //FLOCKSYSTEM_CUH
