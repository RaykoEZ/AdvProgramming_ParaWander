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

/*
/// movement of boid every frame
void Boid::update(const float &_dt)
{
    // test for seek and flee

    FVector desiredV = m_target - m_pos;
    FVector outV = desiredV.GetSafeNormal();
    //FVector f = flee();

    FVector f = seek();

    FVector accel = f * m_invMass;
    FVector oldV = m_v + accel;
    m_v = ClampVector(oldV, FVector(-m_vMax*0.5f), FVector(m_vMax));
    m_pos += m_v;
    m_mesh->SetWorldLocation(m_pos);
    RootComponent->SetWorldLocation(m_pos);
    //SetActorLocation(m_pos);
    //UE_LOG(LogTemp, Warning, TEXT("m_pos : (%f , %f, %f)"), m_pos.X, m_pos.Y, m_pos.Z);
    //UE_LOG(LogTemp, Warning, TEXT("dir : (%f , %f, %f)"), m_v.X, m_v.Y, m_v.Z);


}

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

    FVector desiredV = m_target - m_pos;
    FVector outV = desiredV.GetSafeNormal();
    FVector accel = _f * m_invMass;
    FVector oldV = m_v + accel;

    m_v = ClampVector(oldV, FVector(-m_vMax, -m_vMax, 0.0f), FVector(m_vMax, m_vMax, 0.0f));

    m_pos += m_v;
    /// Update visuals
    m_mesh->SetWorldLocation(m_pos);
    RootComponent->SetWorldLocation(m_pos);
    //
}


/// Seek a position to steer towards
FVector Boid::seek() const
{
    FVector desiredV = m_target - m_pos;
    FVector outV = desiredV.GetSafeNormal();
    if (!FMath::IsNearlyEqual(outV.Size(),100.0f))
    {

        outV *= m_vMax;
        outV -= m_v;

        outV.Z = 0.0f;
        // Draw direction line for debug

        return outV;
    }
    //UE_LOG(LogTemp, Warning, TEXT("boid reached target"));
    return desiredV;
}

FVector Boid::flee()
{
    /// steer away from the seeking position

    FVector desiredV =  m_pos - m_target;
    FVector outV = desiredV.GetSafeNormal();
    if (!FMath::IsNearlyEqual(outV.Size(), 100.0f))
    {

        outV *= m_vMax;
        outV -= m_v;

        outV.Z = 0.0f;
        // Draw direction line for debug

        return outV;
    }
    //UE_LOG(LogTemp, Warning, TEXT("boid reached target"));
    return -m_v;
}



FVector Boid::wander() const
{

    FVector future = m_pos + 10.0f * m_v;
    FVector randRot = FRotator(0.0f, FMath::RandRange(-180.0f, 180.0f), 0.0f).Vector();
    FVector randPos = future + 5.0f * randRot;

    return randPos;
}

FVector Boid::getAverageNeighbourPos(const EBoidType &_t)
{
    TArray<int> idx;
    FVector newP = FVector(0.0f);

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
        //UE_LOG(LogTemp, Warning, TEXT("newP : (%f , %f, %f)"), newP.X, newP.Y, newP.Z);

        //m_target = newP;
        return newP;
    }
    return m_target;
}
*/
