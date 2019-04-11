#ifndef GPUUNITTESTS_H
#define GPUUNITTESTS_H
#include "gtest/gtest.h"
#include <vector>
#include "FlockSystem.h"
#include "DeviceTestKernels.cuh"
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
            FlockSystem flockSys(n,10.0f,0.1f,dt,1.0f,res);
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
            FlockSystem flockSys(n,10.0f,0.1f,dt,1.0f,res);

            flockSys.init();
            flockSys.tick();

            std::vector<uint> hash;
            hash.resize(n);
            flockSys.exportDeviceVector<uint>(hash,flockSys.d_hash);
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
            unsigned int n = 100;
            FlockSystem flockSys(n,10.0f,0.1f,dt,1.0f,res);
            flockSys.init();
            flockSys.tick();

            std::vector<bool> collisionFlag;
            collisionFlag.resize(n);
            flockSys.exportDeviceVector<bool>(collisionFlag,flockSys.d_isThereCollision);
            
        }

        TEST(FlockingTest, RuntimeTest_Boid_Behaviour)
        {

        }


        TEST(FlockingTest, RuntimeTest_Integrator)
        {

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
