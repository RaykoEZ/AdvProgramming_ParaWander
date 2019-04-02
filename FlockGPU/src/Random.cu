#include "Random.cuh"
#include <cuda.h>
#include <curand.h>

void randomFloats(float * &_out, const size_t _n, const uint &_seed)
{
    curandGenerator_t rng;

    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);

    int seed = time(NULL)+ _seed;
    curandSetPseudoRandomGeneratorSeed(rng, seed);

    curandGenerateUniform(rng, _out ,_n);

    curandDestroyGenerator(rng);

}

