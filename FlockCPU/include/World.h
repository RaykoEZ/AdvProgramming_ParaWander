#ifndef WORLD_H
#define WORLD_H
#include "Boid.h"

class World
{

public:
    // Sets default values for this pawn's properties
    World();
    ~World();
    // Called when the game starts or when spawned
    void beginPlay();

public:
    // Called every frame
     void tick(const float &_dt);

    /// @brief initialise simulation objects here
    void initSim();

    ///@brief Radius of the simulation world
    float m_worldRad;
    glm::vec3 m_pos;
};
#endif // WORLD_H
