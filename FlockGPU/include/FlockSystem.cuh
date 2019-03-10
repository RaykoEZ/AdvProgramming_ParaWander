#ifndef FLOCKSYSTEM_CUH
#define FLOCKSYSTEM_CUH
#include "FlockDefines.h"
#include <vector>

class FlockSystem {

public:
    /// Construct an empty fluid system
    FlockSystem();

    /// Destruct our fluid system
    ~FlockSystem();

    /// Initialise a relatively standard dambreak simulation
    void setup(const unsigned int &_numP, const unsigned int &_res);
protected:
    /// Keep track of whether the simulation is ready to start
    bool m_finishedInit;

    void init(const unsigned int &_numP, const unsigned int &_res);

    void clear();
};
#endif //FLOCKSYSTEM_CUH
