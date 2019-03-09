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

#define DEFAULT_MASS 10.0f
#define DEFAULT_MASS_INV 1.0f/DEFAULT_MASS

#endif // FLOCKDEFINES_H
