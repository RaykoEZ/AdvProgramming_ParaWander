#ifndef GPUUNITTESTS_H
#define GPUUNITTESTS_H
#include "gtest/gtest.h"
#include <vector>
#include "FlockSystem.h"

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

    namespace DataSanityTest
    {

    }

    namespace ConfigParamTest
    {

    }
    
    namespace KernelTests
    {

        namespace SpatialHashTest
        {


        }

        namespace AvgNeighbourPosTest
        {

        }

        namespace BoidBehaviourTest
        {

        }

        namespace UtilityTest
        {



        }        

    }

}

#endif // GPUUNITTESTS_H
