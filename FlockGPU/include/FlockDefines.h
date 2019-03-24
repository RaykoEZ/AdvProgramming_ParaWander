#ifndef FLOCKDEFINES_H
#define FLOCKDEFINES_H

// Constants to be used in the sim
#define DEFAULT_TIMESTEP 0.01f

#ifndef CUDART_PI_F
    #define CUDART_PI_F 3.141592654f
#endif

#ifndef CUDART_RADIAN_F
    #define CUDART_RADIAN_F 3.141592654f/180.0f
#endif

/// Define the null hash in case the particle manages to make it's way out of the bounding grid
#define NULL_HASH UINT_MAX

#define DEFAULT_MASS 10.0f
#define DEFAULT_MASS_INV 1.0f/DEFAULT_MASS
#define DEFAULT_CELL_RESOLUTION 100.0f
#define DEFAULT_VMAX 10.0f
#endif // FLOCKDEFINES_H
