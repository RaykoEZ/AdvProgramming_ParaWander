#include "Boid.h"
#include "World.h"
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>


Boid::Boid(const unsigned int &_id,
           const float &_m,
           const glm::vec3 &_pos,
           const glm::vec3 &_v,
           const float &_vMax,
           World *_world)
    :
    m_id(_id),
    m_mass(_m),
    m_invMass(1.0f/_m),
    m_vMax(_vMax),
    m_vMaxDef(_vMax),
    m_world(_world),
    m_pos(_pos),
    m_v(_v)

{

    m_collision = false;
    m_collisionRad = 10.0f;
    m_col = glm::vec3(255.0f);
    m_target = glm::vec3(0.0f);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    m_rng = gen;
}


// Called every frame
void Boid::tick(const float &_dt)
{
    glm::vec3 collisionTarget = getAverageNeighbourPos();
    /// above function returns old m_target if no collision
    glm::vec3 f;
    if(m_collision)
    {
        m_target = collisionTarget;
        f = flee();
        m_col = glm::vec3(255.0f,0.0f,0.0f);

    }
    else
    {
        m_target = wander();
        f = seek();
        m_col = glm::vec3(0.0f,255.0f,0.0f);

    }
    m_collision = false;
    resolve(_dt, f);
}


//------------------------------------------------------------------------


/// Implementations from ideas based on this paper :
/// Steering Behaviors For Autonomous Characters
/// by Craig W.Reynolds, presented on GDC1999

void Boid::resolve(const float &_dt, const glm::vec3 &_f)
{
    /// to do: use a prey boid resolve funcyion from repo!!!!!!!!!!
    glm::vec3 accel = _f * m_invMass;
    glm::vec3 oldV = m_v + accel;
    //std::cout <<"oldV: " << oldV.x <<','<< oldV.y<<'\n';
    //std::cout <<"m_v: " << m_v.x <<','<< m_v.y<<'\n';
    m_v = glm::clamp(oldV, glm::vec3(-m_vMax, -m_vMax, 0.0f), glm::vec3(m_vMax, m_vMax, 0.0f));
    m_v = glm::normalize(m_v);
    m_pos += m_v * _dt;

}


/// Seek a position to steer towards
glm::vec3 Boid::seek() const
{
    glm::vec3 desiredV = m_target - m_pos;
    desiredV = glm::normalize(desiredV);

    if (glm::length(desiredV)!= 0.0f)
    {
        //std::cout <<"Seeking "<<'\n';

        desiredV *= m_vMax;
        desiredV -= m_v;

        return desiredV;
    }
    //std::cout <<"Reached " <<'\n';

    return desiredV;
}

glm::vec3 Boid::flee() const
{
    /// steer away from the seeking position

    glm::vec3 desiredV =  m_pos - m_target;
    desiredV = glm::normalize(desiredV);
    if (glm::length(desiredV)!= 0.0f)
    {

        desiredV *= m_vMax;
        desiredV -= m_v;
        // Draw direction line for debug

        return desiredV;
    }
    return -m_v;
}



glm::vec3 Boid::wander()
{

    std::uniform_real_distribution<float> dis(-180.0f, 180.0f);


    /// get a future direction and randomly generate possible future directions
    glm::vec3 future = m_pos + 10.0f * m_v;

    glm::vec3 randPos = future + glm::rotate((m_v),glm::radians(dis(m_rng)),glm::vec3(0.0f,0.0f,1.0f));
    //glm::vec3 randPos = future + glm::vec3(futureRot,0.0f);
    //std::cout <<"Rand Target: " << randPos.x <<','<< randPos.y<<'\n';
    //std::cout <<"Rand rot: " << dis(m_rng)<<'\n';

    //std::cout <<"Rand Target: " << glm::rotateX(m_v,dis(m_rng)).x<<','<< glm::rotateX(m_v,dis(m_rng)).y <<'\n';
    //m_v = glm::normalize(randPos - m_pos);

    return randPos;
}

glm::vec3 Boid::getAverageNeighbourPos()
{
    unsigned int numNeighbour = 0;
    glm::vec3 newP = glm::vec3(0.0f);
    // find nearby boid index
    for(unsigned int i = 0; i < m_world->m_boids.size(); ++i )
    {
        float dist = glm::distance(m_world->m_boids[i].m_pos,m_pos);
        //summing positions for averaging later
        if(dist <= m_collisionRad && m_world->m_boids[i].m_id != m_id)
        {
            newP +=  m_world->m_boids[i].m_pos;
            ++numNeighbour;
        }
    }
    // get average position of those neighbouring boids
    if (numNeighbour > 0)
    {
        m_collision = true;
        newP /= numNeighbour;
        return newP;
    }
    // passthrough if there are no neighbours
    return m_target;
}

