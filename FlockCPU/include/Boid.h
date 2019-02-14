#ifndef BOID_H
#define BOID_H
#include "glm/vec3.hpp"
#include "glm/glm.hpp"



class Boid
{

public:
    // Sets default values for this actor's properties
    Boid();
    ~Boid();
    /// @brief updates the agent every frame
    /// @param [in] _dt time difference from the previous call
     void update(const float &_dt);
    /// @brief monitors and modifies the states of this agent
     void handleStatus();
    /// @brief function called when an agent enters this neighbourhood
     void onEnterRange();
    /// @brief steers agent towards a target position
    /// @return steering force
    glm::vec3 seek() const;
    /// @brief steers agent away from a target position
    /// @return steering force
    glm::vec3 flee();
    /// @brief applies force to the agent and updates position
    /// @param [in] _force to use
    void resolve(const glm::vec3 &_f);
    /// @brief steer the agent to simulate wandering/grazing
    /// @return random target position to steer towards
    glm::vec3 wander()const; /// d
    /// @brief get the average position of the typed agents in the neighbourhood
    /// @param [in] _t type of agent position to look for
    /// @return a target vector to steer the boid towards to approach/leave a neighbourhood
    //glm::vec3 getAverageNeighbourPos(const EBoidType &_t); ///d

private:
    // Called when the game starts or when spawned
    void beginPlay();
    ///@brief type of boid
    ///@brief general behaviour that determines the flocking behaviour of the agent
    //EBoidStatus m_status;

    ///@brief pointers to all agents in the neighbourhood
    //TArray<Boid*> m_neighbours;

public:
    // Called every frame
     void tick(float _dt);

    /// @brief setter for target position
    /// @param [in] _pos the position of the particle
    void setTarget(const glm::vec3 &_pos) { m_target = _pos; }
    ///@brief whether this agent is out of bound from the meta agent or the eorld sphere
    bool m_isOutOfBound;
    ///@brief type of this agent
    int m_id;
    ///@brief for a = f/m, 1/m pre-calculated
    float m_invMass;
    ///@brief set mesh for a boid
    float m_mass;

    /// @brief max speed gain
    float m_vMax;
    ///@brief max speed gain default
    float m_vMaxDef;
    ///@brief max force gain
    float m_fMax;
    ///@brief radius for the neighbourhood
    float m_collisionRad;
    ///@brief position of boid
    glm::vec3 m_pos;

    ///@brief current velocity
    glm::vec3 m_v;
    ///@brief target position to move to/focus on
    glm::vec3 m_target;





    ///@brief seeded pseudo random number generator for initialisation
    //FRandomStream m_rng;
};
#endif // BOID_H
