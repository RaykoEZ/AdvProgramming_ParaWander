# AdvProgramming_ParaWander
___

 ### This project consists of a CPU version and a GPU version of a few basic flocking behaviours such as: 
- Seeking 
- Fleeing
- Wandering
- Neighbourhood colision detection


___

## CPU Version
___

> ### Design
#### Original API Design for CPU:

The original floacking framework is referenced and improved from my [previous project](https://github.com/RaykoEZ/SimTech_Crowd).

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/FlockCPU/ClassDiagram_CPU.png "ClassDiagram_Old")
---
#### Edited API Design for CPU:

Later on, the framework is refactored to improve organisation and readability. 

The Boid class (from above) contains their own steering methods yet the behaviour of those individual methods are identical. Therefore, it is considered to be better to separate the stated methods into a common namespace for organisation.  

The neighbourhood access is serial (brute force) in this version and the GPU version seeks to improve the overall performance in this area.

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/FlockCPU/ClassDiagram_CPU_Edited.png "ClassDiagram_Old")
___
## GPU Version
___
> ### Design
#### API Design for GPU:

The design for the GPU version separates the concept of "boids" and express them as vector of properties to process; each boids would be defined an index to access the vectors.

Spatial data is also wanted to define and access a neighbourhood from a boid's position in a confined world grid. (General approach inspired by [NCCA/libfluid, Richard Southern](https://github.com/NCCA/libfluid))

This is somewhat different compared to approach of the serial CPU version: 

- As the serial version deals in explicit world positions, the boids don't need to be confined in a set area for tracking.
- However, the use of spatial partitioning limits the boids to only be tracked inside a grid, defining bigger grids may introduce performance hits due to memory limitations in the GPU.  
![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/FlockGPU/ClassDiagram_GPU.png "ClassDiagram_Old")

___
## Setting up FlockCPU/FlockGPU
___

### 1. Both libraries, with the correct dependencies and paths set in their .pro and .pri files, simply input in shell:
1. make clean
2. qmake
3. make

### 2. After successfully building the library, link the library file, preferably using an r-path definition in your project. For more guidance, please check out the [demo files](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/tree/master/Demo).
___
### CPU Version
[Video of running the demo](https://drive.google.com/file/d/1Co5BZrWMo-KVzUiDqZRSK2iAy287XCuu/view?usp=sharing)

#### Dependencies
- This project uses **qmake** to create Makefiles
- C++11 Standard for host dependency
- Using GLM experimental headers, need to define GLM_ENABLE_EXPERIMENTAL
___
### GPU Version
[Video of running the demo](https://drive.google.com/file/d/1Wy13_JAenTxbpbnAPSoZD2hO5POW4xtt/view?usp=sharing)

#### Dependencies
- This project uses **qmake** to create Makefiles
- Using CUDA 9.0
- Currently builds for NVDIA GPUs with architecture of **sm_50**
- C++11 Standard for nvcc and host dependency
- Thrust libraries required to build
> #### For nvcc's CUDA Library inclusion, these flags are used:
>  -lcudart -lcurand -licudata -lcudart_static -lcudadevrt
___

## Performance Comparison

This benchmark measures the length of time for a single frame being produced by the libraries, benchmarks are run on NVDIA Quadro M4400. 

Function timing is taken using Google Benchmark; in two situations:
- Minimal active applications active
- Moderate active applications active

### Average Results:
> ### For Data Table and overhead records, please check [here](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/Table_More.png)

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/Steering.png "Steering")

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/Hash.png "Hashing")

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/GPUTick.png "TickGPU")

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/CPUTick.png "TickCPU")

![](https://github.com/NCCA/advancedprogramming201819-RaykoEZ/blob/master/Benchmarks/Graphs/NeighbourComparison.png "Compare")
- As we can see, serial versions tend to have better timing for task under a specific data size, whilst the parallel version (using the GPU) begins with a sharp overhead (due to memory transfer between host and device) but as the size increases; its timing starts to drastically outperform the serial version inside the CPU. 

### Analysis and Evaluation:

- Within the development stage of neighbourhood search kernel, the thought of utilising shared memory for frequently-used values (such as: positions and collision flags, as they are used more than once for each thread within a block) has been raised. 

> - However, as the number of boids increases for the simulation, each of these values grow linearly with it, therefore, there is a possiblity of shared memory overflow as my grid resolution (number of blocks) increases with the number of boids in consideration of the current GPU having 96KB (48KB per block shown in lab build) of shared mamory for each streaming multiprocessor, which may not be enough for a possible neighbourhood; larger than approximately 3000 boids; assuming that we have *numBoid -1* number of neighbourhood elements needed to be cached in a cell(16B for float3 + 1B for bool for one minimal shared boid data).

- A possible improvement to the state selection can be implemented by replacing the collision flag with a floating point vector for  weighting the resultant forces instead of resorting to creating a branch in a kernel. This improvement can also benefit the extensibility of the system for additional force inclusions from other flocking behaviours for future development.

- Overhead from function calls are neglected within the calculation which may affect the ultimate timing for device functions. 

- Overhead from memeory transfer between host and device are estimated using additional benchmarks (please check table); overhead results are somewhat over-estimated due to additional function overheads.

- In terms of impact of the overhead, neighbourhood process seems to have been somewhat over-estimated as several thrust-based operations (such as sorting and transform algorithms) are needed for the setup, however, they are not needed during the neighbourhood operation in normal use.

- In the case of exporting a flocking system, it would be preferable to have the ability to process multiple frames of the simulation before transferring back to the host. For this project, I have only processed through 1 frame per transfer, this results in the program suffering from the maximum number of overheads as the number of frames are being calculated.

- On the other hand, calculating multiple frames in one parse may introduce memeory limitations as well as a complex data model, which is out of the scope for this project.

- Nevertheless, the general pattern still suggests that - as the number of independent workload increases, GPU processing's advantages outwieghs the overhead of data transfer. 

- However, the growth of the speed-up seems to not completely follow general theories; sources from [Memory Transfer Overhead for CUDA](https://www.cs.virginia.edu/~mwb7w/cuda_support/memory_transfer_overhead.html) would suggest that the overhead for larger data would aggressively counteract against the already-diminishing returns implied in Amdahl's law.
___
## References

1. [Previous Crowd Simulation Project, Chengyan Zhang](https://github.com/RaykoEZ/SimTech_Crowd)

2. [Memory Transfer Overhead for CUDA](https://www.cs.virginia.edu/~mwb7w/cuda_support/memory_transfer_overhead.html)

3. [Benchmarking Guide, Google](https://github.com/google/benchmark)

4. [Nature Of Code - Autonomous Agents, Daniel Shiffman](https://natureofcode.com/book/chapter-6-autonomous-agents/)

5. [NCCA/libfluid, Richard Southern](https://github.com/NCCA/libfluid)

6. [A Comparative Analysis of Spatial Partitioning
Methods for Large-scale, Real-time Crowd
Simulation, Li Bo & Ramakrishnan Mukundan](https://pdfs.semanticscholar.org/346e/69c192d663831b67bca7919b2e10c9a736e2.pdf)

7. [C++ Benchmarks on Vectors, Baptiste Wicht](https://baptiste-wicht.com/posts/2012/12/cpp-benchmark-vector-list-deque.html)

8. [NIVDIA Quadro Catalog, Wikipedia](https://en.wikipedia.org/wiki/Nvidia_Quadro)

9. [Thrust Documentation](https://github.com/thrust/thrust/wiki/Documentation)

10. [Thrust Doxygen](https://thrust.github.io/doc/index.html)
