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

            std::vector<float3> posH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(1.0f,0.5f,0.0f),
                make_float3(-0.5f,0.0f,0.0f)
            };

            std::vector<int3> gridIdxH
            {
                make_int3(0,0,0),
                make_int3(0,0,0),
                make_int3(0,0,0)
            };

            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<int3> gridIdx = gridIdxH;

            testGridFromPoint(gridIdx, pos);

            thrust::copy(gridIdx.begin(), gridIdx.end(), gridIdxH.begin());

            EXPECT_EQ(gridIdxH[0].x, 0);
            EXPECT_EQ(gridIdxH[0].y, 0);
            EXPECT_EQ(gridIdxH[0].z, 0);

            EXPECT_EQ(gridIdxH[1].x, 128);
            EXPECT_EQ(gridIdxH[1].y, 64);
            EXPECT_EQ(gridIdxH[1].z, 0);

            EXPECT_EQ(gridIdxH[2].x, -64);
            EXPECT_EQ(gridIdxH[2].y, 0);
            EXPECT_EQ(gridIdxH[2].z, 0);
        }


        TEST(UtilTest, RuntimeTest_Cell_From_Grid)
        {
            unsigned int n = 1;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);
            flockSys.init();

            /// Test cases: inside the grids, outside and on the boundary
            std::vector<int3> gridIdxH
            {
                make_int3(1,1,0), // second cell on second row
                make_int3(-res,res+1,0), // out of bound
                make_int3(res,res,0) // last cell 

            };

            std::vector<uint> cellIdxH {0,0,0};

            thrust::device_vector<uint> cellIdx = cellIdxH;
            thrust::device_vector<int3> gridIdx = gridIdxH;

            testCellFromGrid(cellIdx, gridIdx);

            thrust::copy(cellIdx.begin(), cellIdx.end(), cellIdxH.begin());

            EXPECT_EQ(cellIdxH[0], 129);
            EXPECT_EQ(cellIdxH[1], NULL_HASH);
            EXPECT_EQ(cellIdxH[2], (res*res) + res);

        }


        TEST(UtilTest, RuntimeTest_Distance_Squared)
        {
            unsigned int n = 1;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);
            flockSys.init();

            std::vector<float3> pos1H
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)
            };

            std::vector<float3> pos2H
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(12.0f,9.0f,0.0f),
                make_float3(-12.0f,-9.0f,0.0f)

            };

            std::vector<float> dist2H {0.0f,0.0f,0.0f};

            thrust::device_vector<float3> pos1 = pos1H;
            thrust::device_vector<float3> pos2 = pos2H;
            thrust::device_vector<float> dist2 = dist2H;
            
            testDist2(dist2, pos1, pos2);
            thrust::copy(dist2.begin(), dist2.end(), dist2H.begin());

            EXPECT_FLOAT_EQ(dist2H[0], 0);
            EXPECT_FLOAT_EQ(dist2H[1], 225);
            EXPECT_FLOAT_EQ(dist2H[2], 225);

        }

        TEST(UtilTest, RuntimeTest_Rotate_Vector_About_Z)
        {
            unsigned int n = 1;
            FlockSystem flockSys(n,10.0f,0.1f,dt,res);
            flockSys.init();


            std::vector<float3> vH
            {
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f)
            };

            std::vector<float3> rotH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),

            };

            std::vector<float> angleH
            {
                0.0f, // 0
                0.125f, // pi/4
                0.25f,
                0.5f,
                1.0f
            };

            thrust::device_vector<float> angle = angleH;
            thrust::device_vector<float3> rot = rotH;
            thrust::device_vector<float3> v = vH;

            testRotateZ(rot, v, angle);

            thrust::copy(rot.begin(), rot.end(), rotH.begin());


            EXPECT_FLOAT_EQ(rotH[0].x, 1.0f);
            EXPECT_FLOAT_EQ(rotH[0].y, 1.0f);
            EXPECT_FLOAT_EQ(rotH[0].z, 0.0f);
                       
            EXPECT_FLOAT_EQ(rotH[1].x, 0.0f);
            EXPECT_FLOAT_EQ(rotH[1].y, 1.414213562f);
            EXPECT_FLOAT_EQ(rotH[1].z, 0.0f);
            
            EXPECT_FLOAT_EQ(rotH[2].x, 1.0f);
            EXPECT_FLOAT_EQ(rotH[2].y, -1.0f);
            EXPECT_FLOAT_EQ(rotH[2].z, 0.0f);
            
            EXPECT_FLOAT_EQ(rotH[3].x, -1.0f);
            EXPECT_FLOAT_EQ(rotH[3].y, -1.0f);
            EXPECT_FLOAT_EQ(rotH[3].z, 0.0f);
            
            EXPECT_FLOAT_EQ(rotH[4].x, 1.0f);
            EXPECT_FLOAT_EQ(rotH[4].y, 1.0f);
            EXPECT_FLOAT_EQ(rotH[4].z, 0.0f);

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
            /// let's make the timestep bigger for testing numbers
            float dtBig = 0.1f;
            float mass = 10.0f;
            FlockSystem flockSys(n,mass,0.1f,dtBig,res);
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
                make_float3(150.0f,90.0f,0.0f),
                make_float3(-150.0f,-90.0f,0.0f)
            };
            float vMax = 12.0f;


            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<float3> v = vH;
            thrust::device_vector<float3> f = fH;

            testResolveForce(pos, v, f, vMax);

            /// copy back
            thrust::copy(pos.begin(),pos.end(),posH.begin());

            /// - new v mag is a zero vector (v + f/m = 0), new pos == pos
            /// - new v mag is non-zero, within min and max of vMax ( -vMax <= v + f/m <= vMax), new pos = pos + norm(v) * dt
            /// - new v mag is non-zero, outside of vMax max-boundary ( -vMax > v + f/m ), new pos = pos + norm(clamp(v)) * dt
            /// - new v mag is non-zero, outside of vMax min-boundary (v + f/m > vMax), new pos = pos + norm(clamp(v)) * dt
            EXPECT_FLOAT_EQ(posH[0].x, 0.0f);
            EXPECT_FLOAT_EQ(posH[0].y, 0.0f);
            EXPECT_FLOAT_EQ(posH[0].z, 0.0f);

            /// f.x = 10 / 10 => 1
            /// newV = (2, 0, 0)
            /// outV = norm(2, 0, 0) => (1, 0, 0)
            /// pos = pos + dt =>  (0.01, 0, 0)

            EXPECT_FLOAT_EQ(posH[1].x, 0.1f);
            EXPECT_FLOAT_EQ(posH[1].y, 0.0f);
            EXPECT_FLOAT_EQ(posH[1].z, 0.0f);

            /// f.x = 150 / 10 => 15 , 90/10 => 9
            /// newV = (16, 9, 0) clamp it
            /// outV = norm(12, 9, 0) => (12/15, 9/15, 0) => (0.8. 0.6, 0)
            /// pos = pos + dt =>  (8e-2, 6e-2, 0)
            EXPECT_FLOAT_EQ(posH[2].x, 0.08f);
            EXPECT_FLOAT_EQ(posH[2].y, 0.06f);
            EXPECT_FLOAT_EQ(posH[2].z, 0.0f);


            /// f.x = -150 / 10 => -15 , -90/10 => -9
            /// newV = (-12, -9, 0) clamp it
            /// outV = norm(-12, -9, 0) => (-0.8. -0.6, 0)
            /// pos = pos + dt =>  (-8e-2, -6e-2, 0)
            EXPECT_FLOAT_EQ(posH[3].x, -0.08f);
            EXPECT_FLOAT_EQ(posH[3].y, -0.06f);
            EXPECT_FLOAT_EQ(posH[3].z, 0.0f);


        }

        TEST(FlockingTest, RuntimeTest_Seek)
        {
            unsigned int n = 1;
            /// let's make the timestep bigger for testing numbers
            float mass = 10.0f;
            FlockSystem flockSys(n,mass,0.1f,dt,res);
            flockSys.init();

            std::vector<float3> fH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };

            std::vector<float3> posH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };

            std::vector<float3> vH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };
            std::vector<float3> targetH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(12.0f,9.0f,0.0f)

            };
            float vMax = 10.0f;
            thrust::device_vector<float3> f = fH;
            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<float3> v = vH;
            thrust::device_vector<float3> target = targetH;

            testSeek(f, pos, v, target, vMax);
            /// copy back
            thrust::copy(f.begin(),f.end(),fH.begin());
            /// target can be either already on boid's position or else where

            EXPECT_FLOAT_EQ(fH[0].x, 0.0f);
            EXPECT_FLOAT_EQ(fH[0].y, 0.0f);
            EXPECT_FLOAT_EQ(fH[0].z, 0.0f);

            /// desiredV = norm(12, 9, 0) = (0.8, 0.6, 0)

            EXPECT_FLOAT_EQ(fH[1].x, 8.0f);
            EXPECT_FLOAT_EQ(fH[1].y, 6.0f);
            EXPECT_FLOAT_EQ(fH[1].z, 0.0f);

        }

        TEST(FlockingTest, RuntimeTest_Flee)
        {
             unsigned int n = 1;
            float mass = 10.0f;
            FlockSystem flockSys(n,mass,0.1f,dt,res);
            flockSys.init();

            std::vector<float3> fH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };
            std::vector<float3> posH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)
            };

            std::vector<float3> vH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };
            std::vector<float3> targetH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(12.0f,9.0f,0.0f)
            };
            float vMax = 10.0f;

            thrust::device_vector<float3> f = fH;
            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<float3> v = vH;
            thrust::device_vector<float3> target = targetH;

            testFlee(f, pos, v, target, vMax);
            /// copy back
            thrust::copy(f.begin(),f.end(),fH.begin());

            EXPECT_FLOAT_EQ(fH[0].x, 0.0f);
            EXPECT_FLOAT_EQ(fH[0].y, 0.0f);
            EXPECT_FLOAT_EQ(fH[0].z, 0.0f);

            EXPECT_FLOAT_EQ(fH[1].x, -8.0f);
            EXPECT_FLOAT_EQ(fH[1].y, -6.0f);
            EXPECT_FLOAT_EQ(fH[1].z, 0.0f);
        }

        TEST(FlockingTest, RuntimeTest_Wander)
        {
            unsigned int n = 1;
            float mass = 10.0f;
            FlockSystem flockSys(n,mass,0.1f,dt,res);
            flockSys.init();

            std::vector<float> angleH
            {
                0.0f, // 0
                0.125f, // pi/4
                0.25f,
                0.5f,
                1.0f
            };
            std::vector<float3> posH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f).
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)

            };

            std::vector<float3> vH
            {
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f),
                make_float3(1.0f,1.0f,0.0f)
            };
            std::vector<float3> targetH
            {
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f),
                make_float3(0.0f,0.0f,0.0f)               
            };

            thrust::device_vector<float> angle = angleH;
            thrust::device_vector<float3> pos = posH;
            thrust::device_vector<float3> v = vH;
            thrust::device_vector<float3> target = targetH;

            testWander(target, angle, v, pos);
            thrust::copy(target.begin(),target.end(),targetH.begin());

            /// future dir = (10, 10, 0)
            /// rotate by 0 => (10, 10, 0)
            /// rotate by pi/4 => (0, 10 x root(2), 0)
            /// rotate by pi/2 => (10, -10, 0)
            /// rotate by pi => (-10, -10, 0)
            /// rotate by 2 pi => (10, 10, 0)

            EXPECT_FLOAT_EQ(targetH[0].x, 10.0f);
            EXPECT_FLOAT_EQ(targetH[0].y, 10.0f);
            EXPECT_FLOAT_EQ(targetH[0].z, 0.0f);
                       
            EXPECT_FLOAT_EQ(targetH[1].x, 0.0f);
            EXPECT_FLOAT_EQ(targetH[1].y, 14.14213562f);
            EXPECT_FLOAT_EQ(targetH[1].z, 0.0f);
            
            EXPECT_FLOAT_EQ(targetH[2].x, 10.0f);
            EXPECT_FLOAT_EQ(targetH[2].y, -10.0f);
            EXPECT_FLOAT_EQ(targetH[2].z, 0.0f);
            
            EXPECT_FLOAT_EQ(targetH[3].x, -10.0f);
            EXPECT_FLOAT_EQ(targetH[3].y, -10.0f);
            EXPECT_FLOAT_EQ(targetH[3].z, 0.0f);
            
            EXPECT_FLOAT_EQ(targetH[4].x, 10.0f);
            EXPECT_FLOAT_EQ(targetH[4].y, 10.0f);
            EXPECT_FLOAT_EQ(targetH[4].z, 0.0f);
        }

    }
}

#endif // GPUUNITTESTS_H
