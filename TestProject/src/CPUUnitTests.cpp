#include "gtest/gtest.h"
#include "World.h"
#include <vector>
#include "glm/vec3.hpp"
/// @file Unit tests for CPU impl are all implemented here:
namespace CPUUnitTests 
{


    /// helper function for testing & diagnose value in a range, learnt from :
    /// https://stackoverflow.com/questions/21972285/expect-a-value-within-a-given-range-using-google-test/21972516
    ::testing::AssertionResult isBetweenInclusive(const float &_val, const float & _min, const float & _max)
    {
        if((_val >= _min) && (_val <= _max))
            return ::testing::AssertionSuccess();
        else
            return ::testing::AssertionFailure()
                   << _val << " is outside the range " << _min << " to " << _max;
    }

    namespace WorldTest
    {
        int one(){ return 1;}
        TEST(one,Sanity)
        {
          EXPECT_EQ(1, one());
        }

        /// World Tests   -------------------------------------------------------------
        TEST(WorldTest, InitializerTest)
        {
            unsigned int n = 1.0;
            float r = 10.0f;
            glm::vec3 p = glm::vec3(0.0f);
            World testWorld =  World(n,r,p);
            EXPECT_EQ(testWorld.m_boids.size(),n);
            EXPECT_FLOAT_EQ(testWorld.m_worldRad, r);

            /// check if the position of boid lies within the spawn range
            EXPECT_TRUE(isBetweenInclusive(testWorld.m_boids[0].m_pos.x,-r,r));
            EXPECT_TRUE(isBetweenInclusive(testWorld.m_boids[0].m_pos.y,-r,r));
            EXPECT_TRUE(isBetweenInclusive(testWorld.m_boids[0].m_pos.z,-r,r));

            EXPECT_FLOAT_EQ(testWorld.m_spawnPos.x,p.x);
            EXPECT_FLOAT_EQ(testWorld.m_spawnPos.y,p.y);
            EXPECT_FLOAT_EQ(testWorld.m_spawnPos.z,p.z);
        }

        TEST(CPU_BoidDataCheck, InitializerList)
        {
           std::vector<glm::vec3> pos;
           std::vector<glm::vec3> col;
           pos.emplace_back(glm::vec3(1.0f));
           col.emplace_back(glm::vec3(1.0f));
           BoidData test = BoidData(pos,col);
           // expect 1.0f in allocated output data, used in World::tick()
           EXPECT_FLOAT_EQ(test.m_col.at(0).x, 1.0f);
           EXPECT_FLOAT_EQ(test.m_col.at(0).y, 1.0f);
           EXPECT_FLOAT_EQ(test.m_col.at(0).z, 1.0f);

           EXPECT_FLOAT_EQ(test.m_pos.at(0).x, 1.0f);
           EXPECT_FLOAT_EQ(test.m_pos.at(0).y, 1.0f);
           EXPECT_FLOAT_EQ(test.m_pos.at(0).z, 1.0f);
        }
        /// Set a target for our test Boid in the world.
        /// Then we would check if the boid is seeking/fleeing from the position
        /// We check the output BoidData with the position and colour of the boid
        TEST(CPU_WorldTest,RuntimeTest)
        {
            unsigned int n = 1.0;
            float r = 10.0f;
            glm::vec3 p = glm::vec3(0.0f);
            World testWorld =  World(n,r,p);

            glm::vec3 target = glm::vec3(10.0f);
            testWorld.m_boids[0].setTarget(target);
            BoidData data = testWorld.tick(1.0f);

            EXPECT_FLOAT_EQ(data.m_col.at(0).x, testWorld.m_boids[0].m_col.x);
            EXPECT_FLOAT_EQ(data.m_col.at(0).y, testWorld.m_boids[0].m_col.y);
            EXPECT_FLOAT_EQ(data.m_col.at(0).z, testWorld.m_boids[0].m_col.z);

            EXPECT_FLOAT_EQ(data.m_pos.at(0).x, testWorld.m_boids[0].m_pos.x);
            EXPECT_FLOAT_EQ(data.m_pos.at(0).y, testWorld.m_boids[0].m_pos.y);
            EXPECT_FLOAT_EQ(data.m_pos.at(0).z, testWorld.m_boids[0].m_pos.z);
        }

    }
    //---------------------------------------------------------------------------------------------------
    namespace BoidTest
    {
        /// Boid Tests -------------------------------------------------------------
        TEST(CPU_BoidTest, InitializerTest)
        {
            World testWorld =  World(1,10.0f,glm::vec3(0.0f));

            unsigned int id = 1;
            float m = 10.0f;
            glm::vec3 pos = glm::vec3(0.0f);
            glm::vec3 v = glm::vec3(0.0f);
            float vMax = 10.0f;
            Boid testBoid = Boid(id, m, pos,v, vMax, &testWorld);

            EXPECT_EQ(testBoid.getId(),id);
            EXPECT_FLOAT_EQ(testBoid.getMass(),10.0f);
            EXPECT_FLOAT_EQ(testBoid.getVMax(),10.0f);
            EXPECT_FLOAT_EQ(testBoid.getVMaxDefault(),10.0f);
            EXPECT_FLOAT_EQ(testBoid.getCollisionRadius(),10.0f);

            EXPECT_FLOAT_EQ(testBoid.getV().x,0.0f);
            EXPECT_FLOAT_EQ(testBoid.getV().y,0.0f);
            EXPECT_FLOAT_EQ(testBoid.getV().z,0.0f);

            EXPECT_FLOAT_EQ(testBoid.getTarget().x,0.0f);
            EXPECT_FLOAT_EQ(testBoid.getTarget().y,0.0f);
            EXPECT_FLOAT_EQ(testBoid.getTarget().z,0.0f);

            EXPECT_FLOAT_EQ(testBoid.m_pos.x,0.0f);
            EXPECT_FLOAT_EQ(testBoid.m_pos.y,0.0f);
            EXPECT_FLOAT_EQ(testBoid.m_pos.z,0.0f);

            EXPECT_EQ(testBoid.getWorld(),&testWorld);
        }

        TEST(CPU_BoidTest, RuntimeTest_Motion_No_Collision)
        {
            World testWorld =  World(1,10.0f,glm::vec3(0.0f));

            float dt = 1.0f;
            Boid old = testWorld.m_boids[0];
            /// take a copy of the boid before updating
            testWorld.tick(dt);

            /// Our boid is wandering right now with nothin gto collide with, check its properties
            EXPECT_FALSE(testWorld.m_boids[0].getCollision());
            /// not rotaing on this axis, so v should not point towards z
            EXPECT_FLOAT_EQ(testWorld.m_boids[0].getV().z,0.0f);
            /// we should have moved
            float dist = glm::distance(old.m_pos,testWorld.m_boids[0].m_pos);
            EXPECT_GT(dist,0.0f);

        }


        TEST(CPU_BoidTest, RuntimeTest_Motion_On_Collision_One_Neighbour)
        {

            World testWorld =  World(1,10.0f,glm::vec3(0.0f));

            glm::vec3 oldP = testWorld.m_boids[0].m_pos;
            unsigned int id = 1;
            float m = 10.0f;
            glm::vec3 pos = glm::vec3(oldP.x,oldP.y,0.0f);
            glm::vec3 v = glm::vec3(0.0f);
            float vMax = 10.0f;
            /// Set the second boid to the position of our first boid in the world
            //Boid testBoid = Boid(id, m, pos,v, vMax, &testWorld);
            /// Add second test boid to the world's list of managed boids
            testWorld.m_boids.push_back(Boid(id, m, pos,v, vMax, &testWorld));

            /// Testing if list is updated correctly for each references of our world
            EXPECT_EQ(testWorld.m_boids.size(),(unsigned int)2);
            EXPECT_EQ(testWorld.m_boids[0].getWorld()->m_boids.size(),(unsigned int)2);
            EXPECT_EQ(testWorld.m_boids[1].getWorld()->m_boids.size(),(unsigned int)2);

            float dist = glm::distance(testWorld.m_boids[0].m_pos,testWorld.m_boids[1].m_pos);
            float r = testWorld.m_boids[0].getCollisionRadius();
            float r1 = testWorld.m_boids[1].getCollisionRadius();
            /// are we colliding? We should be colliding.
            EXPECT_TRUE(isBetweenInclusive(dist,-r,r));
            EXPECT_TRUE(isBetweenInclusive(dist,-r1,r1));

            /// update the two boids in the following order
            testWorld.m_boids[0].tick(0.0f);
            testWorld.m_boids[1].tick(0.0f);

            Boid::NeighbourInfo testInfo1 = testWorld.m_boids[0].getAverageNeighbourPos();
            Boid::NeighbourInfo testInfo2 = testWorld.m_boids[1].getAverageNeighbourPos();

            /// Check if neighbourhood function detects:
            /// - collission
            /// - number of neighbourhood members excluding self
            /// - correct average position of the neighbourhood

            /// There should be a collision
            EXPECT_TRUE(testInfo1.m_isThereCollision);
            EXPECT_TRUE(testInfo2.m_isThereCollision);

            /// They are each other's neighbours, so they would say they'd both have 1 neighbour
            EXPECT_EQ(testInfo1.m_numNeighbour,(unsigned int)1);
            EXPECT_EQ(testInfo1.m_numNeighbour,(unsigned int)1);

            /// if only 1 member in neighbourhood, average = the position of one another
            EXPECT_FLOAT_EQ(testInfo1.m_averagePos.x,testWorld.m_boids[1].m_pos.x);
            EXPECT_FLOAT_EQ(testInfo1.m_averagePos.y,testWorld.m_boids[1].m_pos.y);
            EXPECT_FLOAT_EQ(testInfo1.m_averagePos.z,testWorld.m_boids[1].m_pos.z);

            EXPECT_FLOAT_EQ(testInfo2.m_averagePos.x,testWorld.m_boids[0].m_pos.x);
            EXPECT_FLOAT_EQ(testInfo2.m_averagePos.y,testWorld.m_boids[0].m_pos.y);
            EXPECT_FLOAT_EQ(testInfo2.m_averagePos.z,testWorld.m_boids[0].m_pos.z);

            /// Verifying is collision forwarding is working
            /// First boid would detect second boid for collision as it's updated before second boid
            /// Second would not handle collision if first boid is already moving away
            EXPECT_TRUE(testWorld.m_boids[0].getCollision());
            EXPECT_TRUE(testWorld.m_boids[1].getCollision());




        }

        TEST(CPU_BoidTest, RunTimeTest_On_Collision_More_Than_One_Neighbour)
        {
            World testWorld =  World(1,10.0f,glm::vec3(0.0f));

            glm::vec3 oldP = testWorld.m_boids[0].m_pos;
            unsigned int id = 1;
            float m = 10.0f;
            glm::vec3 pos = glm::vec3(oldP.x,oldP.y,0.0f);
            glm::vec3 v = glm::vec3(0.0f);
            float vMax = 10.0f;
            /// Set the second boid to the position of our first boid in the world
            //Boid testBoid = Boid(id, m, pos,v, vMax, &testWorld);
            /// Add second test boid to the world's list of managed boids
            testWorld.m_boids.push_back(Boid(id, m, pos + 1.0f, v, vMax, &testWorld));
            /// Testing for more than 1 neighbours excluding self
            testWorld.m_boids.push_back(Boid(id+1, m, pos +2.0f, v, vMax, &testWorld));


            std::vector<Boid::NeighbourInfo> testInfo;
            /// update the two boids in the following order
            ///
            /// Check if neighbourhood function detects:
            /// - collission
            /// - number of neighbourhood members excluding self
            /// - correct average position of the neighbourhood
            for(int i = 0; i < 3; ++i)
            {
                testWorld.m_boids[i].tick(0.0f);
                testInfo.push_back(testWorld.m_boids[i].getAverageNeighbourPos());
                EXPECT_TRUE(testInfo[i].m_isThereCollision);
                /// They are each other's neighbours, so they would say they'd both have 1 neighbour
                EXPECT_EQ(testInfo[i].m_numNeighbour,(unsigned int)2);

                /// Verifying is collision forwarding is working
                /// First boid would detect second boid for collision as it's updated before second boid
                /// Second would not handle collision if first boid is already moving away
                EXPECT_TRUE(testWorld.m_boids[i].getCollision());
            }

            /// Get average positions of each boid
            /// 2 members in neighbourhood = (N1 + N2) / 2 for average

            glm::vec3 averageP1 = glm::vec3((testWorld.m_boids[1].m_pos.x + testWorld.m_boids[2].m_pos.x),
                                            (testWorld.m_boids[1].m_pos.y + testWorld.m_boids[2].m_pos.y),
                                            (testWorld.m_boids[1].m_pos.z + testWorld.m_boids[2].m_pos.z)) / 2.0f;

            glm::vec3 averageP2 = glm::vec3((testWorld.m_boids[2].m_pos.x + testWorld.m_boids[0].m_pos.x),
                                            (testWorld.m_boids[2].m_pos.y + testWorld.m_boids[0].m_pos.y),
                                            (testWorld.m_boids[2].m_pos.z + testWorld.m_boids[0].m_pos.z)) / 2.0f;

            glm::vec3 averageP3 = glm::vec3((testWorld.m_boids[1].m_pos.x + testWorld.m_boids[0].m_pos.x),
                                            (testWorld.m_boids[1].m_pos.y + testWorld.m_boids[0].m_pos.y),
                                            (testWorld.m_boids[1].m_pos.z + testWorld.m_boids[0].m_pos.z)) / 2.0f;

            /// Now we verify average position
            EXPECT_FLOAT_EQ(testInfo[0].m_averagePos.x,averageP1.x);
            EXPECT_FLOAT_EQ(testInfo[0].m_averagePos.y,averageP1.y);
            EXPECT_FLOAT_EQ(testInfo[0].m_averagePos.z,averageP1.z);

            EXPECT_FLOAT_EQ(testInfo[1].m_averagePos.x,averageP2.x);
            EXPECT_FLOAT_EQ(testInfo[1].m_averagePos.y,averageP2.y);
            EXPECT_FLOAT_EQ(testInfo[1].m_averagePos.z,averageP2.z);

            EXPECT_FLOAT_EQ(testInfo[2].m_averagePos.x,averageP3.x);
            EXPECT_FLOAT_EQ(testInfo[2].m_averagePos.y,averageP3.y);
            EXPECT_FLOAT_EQ(testInfo[2].m_averagePos.z,averageP3.z);


        }



    }

//-------------------------------------------------------------------------------------------------------

    /// tests for FlockActions' free functions
    namespace SteeringFunctionTest
    {

        TEST(CPU_SteeringTest, Seeking_Test)
        {
            /// Case for pos == target
            glm::vec3 v = FlockFunctions::seek(glm::vec3(0.0f),glm::vec3(0.0f),1.0f,glm::vec3(0.0f));
            EXPECT_FLOAT_EQ(v.x, 0.0f);
            EXPECT_FLOAT_EQ(v.y, 0.0f);
            EXPECT_FLOAT_EQ(v.z, 0.0f);
            /// Case for pos != target
            glm::vec3 expectedV = glm::normalize(glm::vec3(1.0f));
            v = FlockFunctions::seek(glm::vec3(0.0f),glm::vec3(0.0f),1.0f,glm::vec3(1.0f));
            EXPECT_FLOAT_EQ(v.x, expectedV.x);
            EXPECT_FLOAT_EQ(v.y, expectedV.y);
            EXPECT_FLOAT_EQ(v.z, expectedV.z);



        }

        TEST(CPU_SteeringTest, Fleeing_Test)
        {
            /// Case for pos == target
            glm::vec3 v = FlockFunctions::flee(glm::vec3(0.0f),glm::vec3(0.0f),1.0f,glm::vec3(0.0f));
            EXPECT_FLOAT_EQ(v.x, 0.0f);
            EXPECT_FLOAT_EQ(v.y, 0.0f);
            EXPECT_FLOAT_EQ(v.z, 0.0f);

            /// Case for pos != target
            glm::vec3 expectedV = glm::normalize(glm::vec3(-1.0f));
            v = FlockFunctions::flee(glm::vec3(0.0f),glm::vec3(0.0f),1.0f,glm::vec3(1.0f));
            EXPECT_FLOAT_EQ(v.x, expectedV.x);
            EXPECT_FLOAT_EQ(v.y, expectedV.y);
            EXPECT_FLOAT_EQ(v.z, expectedV.z);

        }

        TEST(CPU_SteeringTest, Wander_Test)
        {
            /// Case for zero vectors
            glm::vec3 target = FlockFunctions::wander(glm::vec3(0.0f),glm::vec3(0.0f));

            /// 10.0f is the hardcoded radius of wandering directional selection radius at a future location
            /// If future is stationary due to zero direction vector, we simply search for 10.0f units around us
            EXPECT_TRUE(isBetweenInclusive(target.x,-10.0f, 10.0f));
            EXPECT_TRUE(isBetweenInclusive(target.y, -10.0f,10.0f));
            /// 2d wandering, z ignored
            EXPECT_FLOAT_EQ(target.z, 0.0f);



            /// Now we have initial velocity/direction vector
            ///
            /// Future should be at (10,10,0) with a future radius of 10 with a direction of (1,1,0)
            target = FlockFunctions::wander(glm::vec3(0.0f),glm::vec3(1.0f,1.0f,0.0f));
            EXPECT_TRUE(isBetweenInclusive(target.x,0.0f, 20.0f));
            EXPECT_TRUE(isBetweenInclusive(target.y, 0.0f,20.0f));
            /// 2d wandering, z ignored
            EXPECT_FLOAT_EQ(target.z, 0.0f);

        }

    }

}
