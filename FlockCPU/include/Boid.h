#ifndef BOID_H
#define BOID_H
#include "FlockActions.h"

class World;

class Boid
{
public:
    // Sets default values for this boid's properties
    Boid() = default;

    Boid(const unsigned int &_id,
         const float &_m,
         const glm::vec3 &_pos,
         const glm::vec3 &_v,
         const float &_vMax,
         World *_world);
    ~Boid(){}

    /// @brief data to be passed from the neighbourhood checks,
    /// used for collision detection and flee behaviours
    struct NeighbourInfo
    {
        ///@brief average position of neighbourhood
        glm::vec3 m_averagePos;
        ///@brief number of neighbours in neighbourhood
        unsigned int m_numNeighbour;

        bool m_isThereCollision;

        NeighbourInfo(const glm::vec3 &_p,
                      const unsigned int &_num,
                      const bool &_col)
            : m_averagePos(_p), m_numNeighbour(_num), m_isThereCollision(_col)
        {}
    };


public:

    ///@brief position of boid
    glm::vec3 m_pos;
    ///@brief colour
    glm::vec3 m_col;
    // Called every frame
    void tick(const float &_dt);
    /// @brief setter for target position
    /// @param [in] _pos the position of the particle
    void setTarget(const glm::vec3 &_pos) { m_target = _pos;}

    /// @brief setter for collision detection radius
    /// @param [in] _r radius
    void setCollisionRadius(const float &_r) { m_collisionRad = _r;}

    /// @brief setter for collision detection flag
    /// @param [in] _c whether there is collision or not
    void setCollision(const bool &_c) { m_collision = _c;}
    /// @brief setter for vMax Default
    /// @param [in] _v default value
    void setDefaultVMax(const float &_v) { m_vMaxDef = _v;}
    /// @brief setter for mass
    /// @param [in] _m mass
    void setMass(const float &_m) { m_mass = _m; m_invMass = 1.0f/_m;}


    /// @brief getters
    bool getCollision() const {return m_collision;}
    unsigned int getId() const {return m_id;}
    float getMass() const {return m_mass;}
    float getVMax() const {return m_vMax;}
    float getVMaxDefault() const {return m_vMaxDef;}
    float getCollisionRadius() const {return m_collisionRad;}
    World* getWorld() const {return m_world;}
    glm::vec3 getV() const {return m_v;}
    glm::vec3 getTarget() const {return m_target;}
    /// @brief get the average position of the typed agents in the neighbourhood
    /// @param [in] _t type of agent position to look for
    /// @return a target vector to steer the boid towards to approach/leave a neighbourhood
    NeighbourInfo getAverageNeighbourPos() const;
private:
    ///@brief whether this agent is out of bound from the meta agent or the eorld sphere
    bool m_collision;
    /// @brief id of this boid
    unsigned int m_id;
    ///@brief set mesh for a boid
    float m_mass;
    ///@brief for a = f/m, 1/m pre-calculated
    float m_invMass;
    /// @brief max speed gain
    float m_vMax;
    ///@brief max speed gain default
    float m_vMaxDef;
    ///@brief radius for the neighbourhood
    float m_collisionRad;
    /// @brief pointer ref to its world and properties of world
    World* m_world;
    ///@brief current velocity
    glm::vec3 m_v;
    ///@brief target position to move to/focus on
    glm::vec3 m_target;


};
#endif // BOID_H
