#include "Boid.h"



Boid::Boid()
{


}

// Called when the game starts or when spawned
void Boid::beginPlay()
{
    //m_invMass = 1.0f / m_mass;
    //UE_LOG(LogTemp, Warning, TEXT("m_pos : (%f , %f, %f)"), m_pos.X, m_pos.Y, m_pos.Z);
    //UE_LOG(LogTemp, Warning, TEXT("RootLoc : (%f , %f, %f)"), GetActorLocation().X, GetActorLocation().Y, GetActorLocation().Z);
}


/// movement of boid every frame
void Boid::update(const float &_dt)
{


    glm::vec3 f = seek();

    glm::vec3 accel = f * m_invMass;
    glm::vec3 oldV = m_v + accel;
    m_v = glm::clamp(oldV, glm::vec3(-m_vMax*0.5f), glm::vec3(m_vMax));
    m_pos += m_v * _dt;


}
/*
void Boid::handleStatus()
{
}

void Boid::onEnterRange()
{
}


// Called every frame
void Boid::tick(float DeltaTime)
{
    update(DeltaTime);

}

//------------------------------------------------------------------------


/// Implementations from ideas based on this paper :
/// Steering Behaviors For Autonomous Characters
/// by Craig W.Reynolds, presented on GDC1999

void Boid::resolve(const FVector &_f)
{

    glm::vec3 desiredV = m_target - m_pos;
    glm::vec3 outV = desiredV.GetSafeNormal();
    glm::vec3 accel = _f * m_invMass;
    glm::vec3 oldV = m_v + accel;

    m_v = ClampVector(oldV, glm::vec3(-m_vMax, -m_vMax, 0.0f), glm::vec3(m_vMax, m_vMax, 0.0f));

    m_pos += m_v;
}


/// Seek a position to steer towards
FVector Boid::seek() const
{
    glm::vec3 desiredV = m_target - m_pos;
    desiredV = glm::normalize(desiredV);

    if (!FMath::IsNearlyEqual(desiredV.Size(),100.0f))
    {

        desiredV *= m_vMax;
        desiredV -= m_v;

        outV.Z = 0.0f;
        // Draw direction line for debug

        return desiredV;
    }
    //UE_LOG(LogTemp, Warning, TEXT("boid reached target"));
    return desiredV;
}

glm::vec3 Boid::flee()
{
    /// steer away from the seeking position

    glm::vec3 desiredV =  m_pos - m_target;
    glm::vec3 outV = desiredV.GetSafeNormal();
    if (!FMath::IsNearlyEqual(outV.Size(), 100.0f))
    {

        outV *= m_vMax;
        outV -= m_v;

        outV.Z = 0.0f;
        // Draw direction line for debug

        return outV;
    }
    return -m_v;
}



glm::vec3 Boid::wander() const
{

    glm::vec3 future = m_pos + 10.0f * m_v;
    glm::vec3 randRot = FRotator(0.0f, FMath::RandRange(-180.0f, 180.0f), 0.0f).Vector();
    glm::vec3 randPos = future + 5.0f * randRot;

    return randPos;
}

glm::vec3 Boid::getAverageNeighbourPos(const EBoidType &_t)
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
