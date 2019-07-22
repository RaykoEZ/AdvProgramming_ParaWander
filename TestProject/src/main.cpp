#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "World.h"
#include <vector>
#include "gtest/gtest.h"
#include "FlockSystem.h"
#include "DeviceTestKernels.cuh"
#include "helper_math.h"


int main(int argc, char** argv)
{
    /// Tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
