#include "GeoExporter.h"
#include "World.h"
#include "FlockSystem.h"


int main()
{


    unsigned int n = 10000;
    unsigned int nframes = 300;
    /// CPU VERSION of Flocking Sim
    ///
    ///
/*
    /// dt is higher on CPU due to the implementation using global world space whereas device is contained in a [0,1] region
    float dtH = 0.5f;
    // make our world

    World world = World(n,1000.0f,glm::vec3(0.0f));
    BoidData data;
    for(unsigned int i = 0; i < nframes; ++i)
    {
        std::cout << "Timestep="<<dtH * float(i+1) << "\n";
        data = world.tick(dtH);
        //std::cout << "pos ="<<data.m_pos[0].x<<','<<data.m_pos[0].y << '\n';
        GeoExporter::dumpToGeo(data.m_pos,data.m_col,i);
    }
*/


    /// GPU VERSION
    ///
    float dt = 0.001f;
    float res = 1024.0f;
    FlockSystem flockSys(n,10.0f,0.1f,dt,res);
    flockSys.init();
    std::vector<float3> pos;
    std::vector<float3> col;

    pos.resize(n);
    col.resize(n);

    for(unsigned int i = 0; i < nframes; ++i)
    {
        std::cout << "Timestep="<<dt * float(i+1) << "\n";
        GeoExporter::dumpToGeo(pos,col,i);
        flockSys.tick();
        flockSys.exportResult(pos,col);
    }


    return EXIT_SUCCESS;
}
