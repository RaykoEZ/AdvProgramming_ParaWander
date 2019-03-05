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
    Boid() = default;
    //Boid(const Boid&) = default;
    //Boid& operator=(const Boid&){return *this;}
    //Boid(Boid&&) = default;
    Boid(const unsigned int &_id,
         const float &_m,
         const glm::vec3 &_pos,
         const glm::vec3 &_v,
         const float &_vMax,
         World *_world);
    ~Boid(){}


public:

    ///@brief position of boid
    glm::vec3 m_pos;
    ///@brief colour
    glm::vec3 m_col;
    // Called every frame
    void tick(const float &_dt);
    /// @brief setter for target position
    /// @param [in] _pos the position of the particle
    void setTarget(const glm::vec3 &_pos) { m_target = _pos;}

private:
    ///@brief whether this agent is out of bound from the meta agent or the eorld sphere
    bool m_collision;
    /// @brief id of this boid
    unsigned int m_id;
    ///@brief set mesh for a boid
    float m_mass;
    ///@brief for a = f/m, 1/m pre-calculated
    float m_invMass;
    /// @brief max speed gain
    float m_vMax;
    ///@brief max speed gain default
    float m_vMaxDef;
    ///@brief radius for the neighbourhood
    float m_collisionRad;
    /// @brief pointer ref to its world and properties of world
    World* m_world;
    ///@brief seeded pseudo random number generator for initialisation
    std::mt19937_64 m_rng;
    ///@brief current velocity
    glm::vec3 m_v;
    ///@brief target position to move to/focus on
    glm::vec3 m_target;
    /// @brief steers agent towards a target position
    /// @return steering force
    glm::vec3 seek() const;
    /// @brief steers agent away from a target position
    /// @return steering force
    glm::vec3 flee() const;
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
};
#endif // BOID_H
