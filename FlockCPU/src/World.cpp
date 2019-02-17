#include "World.h"
#include <random>

// Sets default values
World::World(const unsigned int &_nBoids,
             const float &_worldRad,
             const glm::vec3 &_spawnPos):
    m_worldRad(_worldRad),
    m_spawnPos(_spawnPos)
{
    // spawn and add boids into the sim
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> spawnDis(-_worldRad, _worldRad);
    m_boids.reserve(_nBoids);
    for(unsigned int i = 0; i < _nBoids; ++i)
    {
        glm::vec3 pos = m_spawnPos + glm::vec3(spawnDis(gen),spawnDis(gen), 0.0f);
        glm::vec3 v = glm::vec3(0.0f, 0.0f, 0.0f);
        /// Spawning a boid
        /// Get a boid initialized
        //auto boid = std::make_unique<Boid>(10.0f,pos,v,1.0f,1.0f,*this);
        m_boids.push_back(std::make_unique<Boid>(10.0f,pos,v,1.0f,1.0f,this));
        m_boids[i]->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
    }
}


// Called every frame
/// Remember to use timer for simulation-----------------------------------------------
BoidData World::tick(const float &_dt)
{
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> col;
    for(unsigned int i = 0; i < m_boids.size(); ++i )
    {
        m_boids[i]->tick(_dt);
        pos.push_back(m_boids[i]->m_pos);
        col.push_back(glm::vec3(255.0f,0.0f,0.0f));
    }
    return BoidData(pos,col);
}




