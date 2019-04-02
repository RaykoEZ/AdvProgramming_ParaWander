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

}


// Called every frame
void Boid::tick(const float &_dt)
{
    /// 0 mass 0 work
    if(m_mass <= 0.0f) return;

    /// checking for neighbourhood and collision
    NeighbourInfo collisionTarget = getAverageNeighbourPos();
    setCollision(collisionTarget.m_isThereCollision);
    /// collision afftects behaviour or forces applied
    glm::vec3 f;
    if(m_collision)
    {
        m_target = collisionTarget.m_averagePos;
        f = FlockFunctions::flee(m_pos,m_v,m_vMax,m_target);
        m_col = glm::vec3(255.0f,0.0f,0.0f);

    }
    else
    {
        m_target = FlockFunctions::wander(m_pos,m_v);
        f = FlockFunctions::seek(m_pos,m_v,m_vMax,m_target);
        m_col = glm::vec3(0.0f,255.0f,0.0f);

    }

    // we resolve our forces here
    glm::vec3 accel = f * m_invMass;
    // change boid direction, towards the direction of our force
    m_v = f;
    glm::vec3 oldV = m_v + accel;
    //std::cout <<"oldV: " << oldV.x <<','<< oldV.y<<'\n';
    //std::cout <<"m_v: " << m_v.x <<','<< m_v.y<<'\n';
    // if direction/velocity vector is 0, we don't normalize into nans
    if(glm::length(m_v) > 0.0f)
    {
        m_v = glm::clamp(oldV, glm::vec3(-m_vMax, -m_vMax, 0.0f), glm::vec3(m_vMax, m_vMax, 0.0f));
        m_v = glm::normalize(m_v);
    }
    // update boid position
    m_pos += m_v * _dt;

}


//------------------------------------------------------------------------


/// Implementations from ideas based on this paper :
/// Steering Behaviors For Autonomous Characters
/// by Craig W.Reynolds, presented on GDC1999

Boid::NeighbourInfo Boid::getAverageNeighbourPos() const
{
    unsigned int numNeighbour = 0;
    glm::vec3 newP = glm::vec3(0.0f);
    // find nearby boid index
    for(unsigned int i = 0; i < m_world->m_boids.size(); ++i )
    {

        float dist = glm::distance(m_world->m_boids[i].m_pos,m_pos);
        //std::cout<< m_world->m_boids[i].m_v.x<<m_world->m_boids[i].m_v.y << " " << m_id<< " on "<< i <<'\n';
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
        newP /= numNeighbour;
        return NeighbourInfo(newP, numNeighbour, true);
    }
    else
    {
        return NeighbourInfo(m_target, 0, false);
    }

}




