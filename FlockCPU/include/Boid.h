#ifndef BOID_H
#define BOID_H
#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include <random>
#include <memory>

class World;



class Boid
{
public:
    // Sets default values for this boid's properties
    Boid() = delete;
    Boid(const float &_m,
         const glm::vec3 &_pos,
         const glm::vec3 &_v,
         const float &_vMax,
         const float &_fMax,
         World *_world);
    ~Boid();

    /// @brief steers agent towards a target position
    /// @return steering force
    glm::vec3 seek() const;
    /// @brief steers agent away from a target position
    /// @return steering force
    glm::vec3 flee();
    /// @brief applies force to the agent and updates position
    /// @param [in] _force to use
    void resolve(const float &_dt, const glm::vec3 &_f);
    /// @brief steer the agent to simulate wandering/grazing
    /// @return random target position to steer towards
    glm::vec3 wander(); /// d
    /// @brief get the average position of the typed agents in the neighbourhood
    /// @param [in] _t type of agent position to look for
    /// @return a target vector to steer the boid towards to approach/leave a neighbourhood
    glm::vec3 getAverageNeighbourPos();

public:
    // Called every frame
     void tick(const float &_dt);

    /// @brief setter for target position
    /// @param [in] _pos the position of the particle
    void setTarget(const glm::vec3 &_pos) { m_target = _pos; }
    ///@brief whether this agent is out of bound from the meta agent or the eorld sphere
    bool m_isOutOfBound;
    ///@brief set mesh for a boid
    float m_mass;
    ///@brief for a = f/m, 1/m pre-calculated
    float m_invMass;
    /// @brief max speed gain
    float m_vMax;
    ///@brief max speed gain default
    float m_vMaxDef;
    ///@brief max force gain
    float m_fMax;
    ///@brief radius for the neighbourhood
    float m_collisionRad;
    ///@brief position of boid
    glm::vec3 m_pos;
    ///@brief current velocity
    glm::vec3 m_v;
    ///@brief target position to move to/focus on
    glm::vec3 m_target;

private:

    /// @brief pointer ref to its world and properties of world
    World* m_world;
    ///@brief seeded pseudo random number generator for initialisation
    std::mt19937 m_rng;
};
#endif // BOID_H
