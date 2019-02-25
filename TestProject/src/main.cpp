#include <iostream>
#include "gtest/gtest.h"

#ifndef GTEST_SAMPLES_SAMPLE1_H_
#define GTEST_SAMPLES_SAMPLE1_H_

// Returns n! (the factorial of n).  For negative n, n! is defined to be 1.
int one();


#endif // GTEST_SAMPLES_SAMPLE1_H_
int one(){ return 1;}

namespace
{
    TEST(one,Trivial)
    {
      // This test is named "Negative", and belongs to the "FactorialTest"
      // test case.
      EXPECT_EQ(1, one());
    }
}

int main()
{
    RUN_ALL_TESTS();
    return 0;
}
