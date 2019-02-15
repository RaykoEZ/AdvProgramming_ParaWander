#include "Boid.h"
#include <glm/gtx/rotate_vector.hpp>


Boid::Boid(const float &_m,
           const glm::vec3 &_pos,
           const glm::vec3 &_v,
           const float &_vMax,
           const float &_fMax)
    :
    m_mass(_m),
    m_invMass(1.0f/_m),
    m_vMaxDef(_vMax),
    m_fMax(_fMax),
    m_pos(_pos),
    m_v(_v)
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
    glm::vec3 f = seek();

    glm::vec3 accel = f * m_invMass;
    glm::vec3 oldV = m_v + accel;
    m_v = glm::clamp(oldV, glm::vec3(-m_vMax*0.5f), glm::vec3(m_vMax));
    m_pos += m_v * _dt;
}



void Boid::handleStatus()
{
}

void Boid::onEnterRange()
{
}



//------------------------------------------------------------------------


/// Implementations from ideas based on this paper :
/// Steering Behaviors For Autonomous Characters
/// by Craig W.Reynolds, presented on GDC1999

void Boid::resolve(const glm::vec3 &_f)
{
    /// to do: use a prey boid resolve funcyion from repo!!!!!!!!!!
    glm::vec3 accel = _f * m_invMass;
    glm::vec3 oldV = m_v + accel;

    m_v = glm::clamp(oldV, glm::vec3(-m_vMax, -m_vMax, 0.0f), glm::vec3(m_vMax, m_vMax, 0.0f));

    m_pos += m_v;
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
    std::uniform_real_distribution<> dis(-180.0, 180.0);
    /// get a future direction and randomly generate possible future directions
    glm::vec3 future = m_pos + 10.0f * m_v;
    glm::vec2 futureRot = glm::vec2(future.x,future.y);
    futureRot =  0.5f * glm::rotate(futureRot,static_cast<float>(dis(m_rng)));
    glm::vec3 randPos = future + glm::vec3(futureRot,0.0f);

    return randPos;
}
/*
glm::vec3 Boid::getAverageNeighbourPos()
{
    TArray<int> idx;
    glm::vec3 newP = FVector(0.0f);

    if (_t == EBoidType::PREDATOR)
    {
        idx = searchPredator();

    }
    else
    {
        idx = searchPrey();

    }

    if (idx.Num() > 0)
    {

        for (int i = 0; i < idx.Num(); ++i)
        {
            newP += m_neighbours[idx[i]]->m_pos;
        }
        newP /= idx.Num();

        //m_target = newP;
        return newP;
    }
    return m_target;
}
*/
