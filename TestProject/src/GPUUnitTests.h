#ifndef GPUUNITTESTS_H
#define GPUUNITTESTS_H
#include "gtest/gtest.h"
#include <vector>
#include "DeviceTestKernels.cuh"

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

    namespace ConfigParamTest
    {
        /// Testing for valid config parameters for kernels to run
        TEST(ConfigParamTest, InitializerTest)
        {

        }
    }
    
    /// Test for device and global kernels
    namespace KernelTest
    {

        TEST(KernelTest, RuntimeTest_Hash)
        {

        }


        namespace FlockingTest
        {

            TEST(FlockingTest, RuntimeTest_Neighbourhood)
            {

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

        namespace UtilTest
        {

            TEST(UtilTest, RuntimeTest_Grid_From_Pos)
            {


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




    }

}

#endif // GPUUNITTESTS_H
