#include "Random.cuh"
#include <cuda.h>
#include <curand.h>

void randomFloats(float *&_out, const size_t _n)
{
    curandGenerator_t rng;

    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(rng, time(NULL));

    curandGenerateUniform(rng, _out ,_n);

    curandDestroyGenerator(rng);

}

