#ifndef WORLD_H
#define WORLD_H
class  World
{


public:
    // Sets default values for this pawn's properties
    World();
    ~World();
    // Called when the game starts or when spawned
    void beginPlay();

public:
    // Called every frame
     void tick(const float &_dt);

    /// @brief initialise simulation objects here
    void initSim();

    ///@brief Radius of the simulation world
    float m_worldRad;

};
#endif // WORLD_H
