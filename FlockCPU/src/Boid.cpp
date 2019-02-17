#include "Boid.h"
#include "World.h"
#include <glm/gtx/rotate_vector.hpp>


Boid::Boid(const float &_m,
           const glm::vec3 &_pos,
           const glm::vec3 &_v,
           const float &_vMax,
           const float &_fMax,
           World *_world)
    :
    m_mass(_m),
    m_invMass(1.0f/_m),
    m_vMaxDef(_vMax),
    m_fMax(_fMax),
    m_pos(_pos),
    m_v(_v),
    m_world(_world)
{

    m_isOutOfBound = false;
    m_collisionRad = 10.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    m_rng = gen;
}


// Called every frame
void Boid::tick(const float &_dt)
{
    m_target = wander();
    glm::vec3 f = seek();
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

    m_v = glm::clamp(oldV, glm::vec3(-m_vMax, -m_vMax, 0.0f), glm::vec3(m_vMax, m_vMax, 0.0f));

    m_pos += m_v * _dt;

}


/// Seek a position to steer towards
glm::vec3 Boid::seek() const
{
    glm::vec3 desiredV = m_target - m_pos;
    desiredV = glm::normalize(desiredV);

    if (glm::length(desiredV)!= 0.0f)
    {

        desiredV *= m_vMax;
        desiredV -= m_v;

        return desiredV;
    }
    //UE_LOG(LogTemp, Warning, TEXT("boid reached target"));
    return desiredV;
}

glm::vec3 Boid::flee()
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
    glm::vec2 futureRot = glm::vec2(future.x,future.y);
    futureRot =  0.5f * glm::rotate(futureRot,dis(m_rng));
    glm::vec3 randPos = future + glm::vec3(futureRot,0.0f);

    return randPos;
}

glm::vec3 Boid::getAverageNeighbourPos()
{
    unsigned int numNeighbour = 0;
    glm::vec3 newP = glm::vec3(0.0f);
    // find nearby boid index
    for(unsigned int i = 0; i < m_world->m_boids.size(); ++i )
    {
        float dist = glm::distance(m_world->m_boids[i]->m_pos,m_pos);
        //summing positions for averaging later
        if(dist <= m_collisionRad)
        {
            newP +=  m_world->m_boids[i]->m_pos;
            ++numNeighbour;
        }
    }
    // get average position of those neighbouring boids
    if (numNeighbour > 0)
    {
        newP /= numNeighbour;
        return newP;
    }
    // passthrough if there are no neighbours
    return m_target;
}

