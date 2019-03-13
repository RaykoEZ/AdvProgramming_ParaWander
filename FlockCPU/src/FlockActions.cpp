#include "FlockActions.h"
#include <glm/gtx/rotate_vector.hpp>


glm::vec3 FlockFunctions::seek(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target)
{
    glm::vec3 desiredV = _target - _pos;

    if (glm::length(desiredV) > 0.0f)
    {
        //std::cout <<"Seeking "<<'\n';
        desiredV = glm::normalize(desiredV);
        desiredV *= _vMax;
        desiredV -= _v;

        return desiredV;
    }
    //std::cout <<"Reached " <<'\n';

    return desiredV;

}

glm::vec3 FlockFunctions::flee(const glm::vec3 &_pos, const glm::vec3 &_v,const float &_vMax, const glm::vec3 &_target)
{
    /// steer away from the seeking position

    glm::vec3 desiredV =  _pos - _target;
    if (glm::length(desiredV)> 0.0f)
    {
        desiredV = glm::normalize(desiredV);
        desiredV *= _vMax;
        desiredV -= _v;
        // Draw direction line for debug

        return desiredV;
    }
    return -_v;

}
glm::vec3 FlockFunctions::wander(const glm::vec3 &_pos, const glm::vec3 &_v)
{
    static std::mt19937_64 m_rng;
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    m_rng = gen;

    std::uniform_real_distribution<float> dis(-180.0f, 180.0f);


    /// get a future direction and randomly generate possible future directions
    glm::vec3 future = _pos + 10.0f * _v;

    glm::vec3 randPos = future + glm::rotate((_v),glm::radians(dis(m_rng)),glm::vec3(0.0f,0.0f,1.0f));


    return randPos;
}

