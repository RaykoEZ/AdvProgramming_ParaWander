#ifndef GPUUNITTESTS_H
#define GPUUNITTESTS_H
#include "gtest/gtest.h"
#include <vector>
#include "FlockSystem.h"
#include "DeviceTestKernels.cuh"
#include "helper_math.h"

/// @file Unit tests for GPU impl are all implemented here:

namespace GPUUnitTests
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
    




    /// Test for device and global kernels


    namespace UtilTest
    {
        float dt = 0.001f;
        float res = 128.0f;

        
        TEST(UtilTest, RuntimeTest_Grid_From_Pos)
        {
            unsigned int n = 1;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);
            flockSys.init();
            flockSys.tick();



        }
        TEST(UtilTest, RuntimeTest_Cell_From_Grid)
        {


        }
        TEST(UtilTest, RuntimeTest_Distance_Squared)
        {


        }
        TEST(UtilTest, RuntimeTest_Rotate_Vector_About_Z)
        {

        }

    }


    namespace FlockingTest
    {
        float dt = 0.001f;
        float res = 128.0f;

        TEST(FlockingTest, RuntimeTest_Hashing_and_Boid_Spawn)
        {
            uint n = 100;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);

            flockSys.init();
            flockSys.tick();

            std::vector<uint> hash = flockSys.getHash();
            /// test if all boids have valid hash upon spawning
            bool boidInRange = true;
            for(uint i = 0 ;i < n ; ++i)
            {
                if(hash[i] == NULL_HASH)
                {
                    boidInRange = false;
                }
            }
            EXPECT_TRUE(boidInRange);


        }
        TEST(FlockingTest, RuntimeTest_Neighbourhood_and_collision)
        {
            unsigned int n = 1000;
            uint numCollision = 0;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);

            std::vector<float3> colourFlag;
            std::vector<bool> collisionFlag;
            std::vector<uint> collidingBoidIdx;
            std::vector<uint> otherBoidIdx;
            while (numCollision == 0)
            {
                otherBoidIdx.clear();
                flockSys.init();
                flockSys.tick();
                collisionFlag = flockSys.getCollisionFlag();
                for(uint i = 0; i < collisionFlag.size(); ++i)
                {
                    if(collisionFlag[i])
                    {
                        ++numCollision;
                        collidingBoidIdx.push_back(i);

                    }
                    else
                    {
                        otherBoidIdx.push_back(i);
                    }
                }
            }
            // check if there are more than 1 boid in a collision event
            EXPECT_GT(numCollision, 1);
            colourFlag = flockSys.getColour();

            // check for correct colour flags
            for(uint i =0; i< numCollision; ++i)
            {
                EXPECT_FLOAT_EQ(colourFlag[collidingBoidIdx[i]].x, 255.0f);
                EXPECT_FLOAT_EQ(colourFlag[collidingBoidIdx[i]].y, 0.0f);
                EXPECT_FLOAT_EQ(colourFlag[collidingBoidIdx[i]].z, 0.0f);

            }

            for(uint i = 0; i < otherBoidIdx.size(); ++i )
            {
                EXPECT_FLOAT_EQ(colourFlag[otherBoidIdx[i]].x, 0.0f);
                EXPECT_FLOAT_EQ(colourFlag[otherBoidIdx[i]].y, 255.0f);
                EXPECT_FLOAT_EQ(colourFlag[otherBoidIdx[i]].z, 0.0f);

            }

            
        }

        TEST(FlockingTest, RuntimeTest_Integrator)
        {
            unsigned int n = 1;
            float mass = 10.0f;
            FlockSystem flockSys(n,mass,0.1f,dt,res);
            flockSys.init();

            /// test cases:
            /// - new v mag is a zero vector (v + f/m = 0), new pos == pos
            /// - new v mag is non-zero, within min and max of vMax ( -vMax <= v + f/m <= vMax), new pos = pos + norm(v) * dt
            /// - new v mag is non-zero, outside of vMax max-boundary ( -vMax > v + f/m ), new pos = pos + norm(clamp(v)) * dt
            /// - new v mag is non-zero, outside of vMax min-boundary (v + f/m > vMax), new pos = pos + norm(clamp(v)) * dt
            std::vector<float3> posH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)
            };

            std::vector<float3> vH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(1.0f,0.0f,0.0f),
                make_float3(1.0f,0.0f,0.0f),
                make_float3(-1.0f,0.0f,0.0f)

            };
            std::vector<float3> fH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(10.0f,0.0f,0.0f),
                make_float3(110.0f,0.0f,0.0f),
                make_float3(-110.0f,0.0f,0.0f)
            };
            float vMax = 10.0f;


            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<float3> v = vH;
            thrust::device_vector<float3> f = fH;

            testResolveForce(pos, v, f, vMax);

            thrust::copy(pos.begin(),pos.end(),posH.begin());
            thrust::copy(v.begin(),v.end(),vH.begin());
            thrust::copy(f.begin(),f.end(),fH.begin());


        }
        TEST(FlockingTest, RuntimeTest_Seek)
        {

        }
        TEST(FlockingTest, RuntimeTest_Flee)
        {

        }
        TEST(FlockingTest, RuntimeTest_Wander)
        {

        }

    }

}

#endif // GPUUNITTESTS_H
