#ifndef UNITTESTS_H
#define UNITTESTS_H
#include "gtest/gtest.h"
#include "World.h"
#include <vector>
#include "glm/vec3.hpp"
// test
namespace
{
    int one(){ return 1;}
    TEST(one,Sanity)
    {
      // This test is named "Negative", and belongs to the "FactorialTest"
      // test case.
      EXPECT_EQ(1, one());
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

    /// World Tests   -------------------------------------------------------------
    TEST(WorldTest, InitializeTest)
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


    TEST(WorldTest,RuntimeTest)
    {


    }

    /// Boid Tests -------------------------------------------------------------
    TEST(BoidTest, InitializeTest)
    {
        World testWorld =  World(1,10,glm::vec3(0.0f));
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

    TEST(BoidTest, RuntimeTest)
    {

    }

}


#endif //UNITTEST_H
