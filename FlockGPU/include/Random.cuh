#ifndef RANDOM_CUH
#define RANDOM_CUH

///@biref generate random floats
///@param [in/out] _out memory to write to, must be pre-allocated
///@param [in] _n number of floats to generate
///@param [in] _seed seed for rng
///@return normalised to be between 0 and 1, simple use case of curandGenerateUniform
void randomFloats(float *&_out, const size_t _n, const uint &_seed);


#endif // RANDOM_CUH