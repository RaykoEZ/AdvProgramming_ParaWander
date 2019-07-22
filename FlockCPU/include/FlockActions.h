#ifndef FLOCKACTIONS_H
#define FLOCKACTIONS_H

#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include <random>
#include <memory>
///@file Storage of flock behaviours
namespace FlockFunctions
{
    /// @brief steers agent towards a target position
    /// @param [in] _pos position of the boid
    /// @param [in] _v velocity vector of the boid
    /// @param [in] _vMax velocity limit of the boid
    /// @param [in] _target target position of the boid
    /// @return steering force
    glm::vec3 seek(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target);
    /// @brief steers agent away from a target position
    /// @param [in] _pos position of the boid
    /// @param [in] _v velocity vector of the boid
    /// @param [in] _vMax velocity limit of the boid
    /// @param [in] _target target position of the boid
    /// @return steering force
    glm::vec3 flee(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target);
    /// @brief steer the agent to simulate wandering/grazing
    /// @param [in] _pos position of the boid
    /// @param [in] _v velocity vector of the boid
    /// @return random target position to steer towards
    glm::vec3 wander(const glm::vec3 &_pos, const glm::vec3 &_v); /// d

}


#endif //FLOACKACTIONS_H
