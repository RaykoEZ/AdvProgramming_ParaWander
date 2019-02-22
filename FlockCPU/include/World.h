#ifndef WORLD_H
#define WORLD_H
#include "Boid.h"
#include <vector>


struct BoidData
{
    /// @brief position of particle/boid to export
    std::vector<glm::vec3> m_pos;
    /// @brief colour of particle/boid to export
    std::vector<glm::vec3> m_col;
    BoidData(){}
    /// @brief ctor for output data
    BoidData(const std::vector<glm::vec3> &_pos,
             const std::vector<glm::vec3> &_col) : m_pos(_pos), m_col(_col){}
};

class World
{

public:
    // Sets default values for this pawn's properties
    World()=delete;
    World(const unsigned int &_nBoids,
          const float &_worldRad,
          const glm::vec3 &_spawnPos);

    ~World(){}

public:
    // Called every frame
    BoidData tick(const float &_dt);
    /// @brief Radius of the simulation world
    float m_worldRad;
    /// @brief Position to spawn voids
    glm::vec3 m_spawnPos;
    /// @brief container of all boids in the flock
    std::vector<Boid> m_boids;
};
#endif // WORLD_H
