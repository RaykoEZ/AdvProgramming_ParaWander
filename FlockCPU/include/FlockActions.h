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
    /// @return steering force
    glm::vec3 seek(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target);
    /// @brief steers agent away from a target position
    /// @return steering force
    glm::vec3 flee(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target);
    /// @brief steer the agent to simulate wandering/grazing
    /// @return random target position to steer towards
    glm::vec3 wander(const glm::vec3 &_pos, const glm::vec3 &_v); /// d

}


#endif //FLOACKACTIONS_H
