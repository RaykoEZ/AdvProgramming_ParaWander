#ifndef FLOCKSYSTEM_H
#define FLOCKSYSTEM_H
#include "FlockDefines.h"
#include <vector>
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
                const float &_rad,
                const float &_res);

    /// Destruct our fluid system
    ~FlockSystem();
    void init();
    void tick();
    /// copy device results to host for houdini .geo export
    void exportResult(std::vector<float3> &_posh, std::vector<float3> &_colh) const;
    /// Export memebers to host for debugging and testing
    template<typename T>
    void exportDeviceVector(std::vector<T> &_out, const thrust::device_vector<T> &_in) const;



    uint getBlockSize() const { return h_blockSize; }
    dim3 getGridSize() const { return h_gridSize; }



        ///
    thrust::device_vector<bool> d_isThereCollision;
    thrust::device_vector<uint> d_hash;
    thrust::device_vector<uint> d_cellOcc;
    thrust::device_vector<uint> d_scatterAddress;
    thrust::device_vector<float> d_vMax;
    /// @brief a random number between 0 and 1 for steering angle
    thrust::device_vector<float> d_angle;
    thrust::device_vector<float3> d_col;
    thrust::device_vector<float3> d_pos;
    thrust::device_vector<float3> d_v;
    thrust::device_vector<float3> d_target;
protected:

    void clear();
    void prepareBoids(const float &_nBoids,
                      const float &_minX, const float &_minY,
                      const float &_maxX, const float &_maxY);
    /// Keep track of whether the simulation is ready to start
    bool h_init;
    uint h_frameCount;

    uint h_blockSize;
    dim3 h_gridSize;


private:
    FlockParams* h_params;

};

#endif //FLOCK_SYSTEM_H
