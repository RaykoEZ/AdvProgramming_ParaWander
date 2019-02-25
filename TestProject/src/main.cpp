#include <iostream>
#include "gtest/gtest.h"

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
