#ifndef UNITTESTS_H
#define UNITTESTS_H
#include "gtest/gtest.h"
#include "World.h"
#include <vector>
#include "glm/vec3.hpp"
/// @file Unit tests are all implemented here:
namespace UnitTests {


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
          // This test is named "Negative", and belongs to the "FactorialTest"
          // test case.
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

        TEST(BoidDataCheck, InitializerList)
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
        TEST(WorldTest,RuntimeTest)
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
    namespace BoidTest
    {

        World testWorld =  World(1,10,glm::vec3(0.0f));
        /// Boid Tests -------------------------------------------------------------
        TEST(BoidTest, InitializerTest)
        {

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

        TEST(BoidTest, RuntimeTest_Motion_No_Collision)
        {
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


        TEST(BoidTest, RuntimeTest_Motion_On_Collision)
        {

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

            /// First boid would detect second boid for collision as it's updated before second boid
            /// Second would not handle collision if first boid is already moving away
            EXPECT_TRUE(testWorld.m_boids[0].getCollision());
            EXPECT_TRUE(testWorld.m_boids[1].getCollision());

        }

    }


}



#endif //UNITTESTS_H
