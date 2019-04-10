#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "World.h"
#include "CPUUnitTests.h"
#include "GPUUnitTests.h"
/// Dump our data from the fluid problem to a file
/// Taken from: https://github.com/NCCA/libfluid/blob/master/test/src/main.cpp
/// By Richard Southern
void dumpToGeo(const std::vector<glm::vec3> &points,
               const std::vector<glm::vec3> &colour,
               const unsigned int cnt) {
    char fname[100];
    std::sprintf(fname,"geo/flock.%04d.geo", cnt);

    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open()) {
        std::cout << "failed to Open file "<<fname<<'\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << points.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    std::vector<glm::vec3>::const_iterator pit,vit;
    for(pit = points.begin(), vit = colour.begin(); pit != points.end(); ++pit, ++vit)
    {
        // Write out the point coordinates and a "1" (not sure what this is for)
        ss<< (*pit).x <<" "<< (*pit).y <<" "<< (*pit).z << " 1 ";
        // Output the colour attribute (lets leave it white)
        ss<<"("<< (*vit).x <<" "<< (*vit).y <<" "<< (*vit).z <<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<points.size()<<" ";
    for(size_t i=0; i<points.size(); ++i)
    {
      ss<<i<<" ";
    }
    ss<<" [0	0]\n";
    ss<<"box_object1 unordered\n";
    ss<<"1 1\n";
    ss<<"beginExtra\n";
    ss<<"endExtra\n";
    // dump string stream to disk;
    file<<ss.rdbuf();
    file.close();
}
/// GPU output using float3
void dumpToGeo(const std::vector<float3> &points,
               const std::vector<float3> &colour,
               const unsigned int cnt) {
    char fname[100];
    std::sprintf(fname,"geo/flock.%04d.geo", cnt);

    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open()) {
        std::cout << "failed to Open file "<<fname<<'\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << points.size() << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    std::vector<float3>::const_iterator pit,vit;
    for(pit = points.begin(), vit = colour.begin(); pit != points.end(); ++pit, ++vit)
    {
        // Write out the point coordinates and a "1" (not sure what this is for)
        ss<< (*pit).x <<" "<< (*pit).y <<" "<< (*pit).z << " 1 ";
        // Output the colour attribute (lets leave it white)
        ss<<"("<< (*vit).x <<" "<< (*vit).y <<" "<< (*vit).z <<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<points.size()<<" ";
    for(size_t i=0; i<points.size(); ++i)
    {
      ss<<i<<" ";
    }
    ss<<" [0	0]\n";
    ss<<"box_object1 unordered\n";
    ss<<"1 1\n";
    ss<<"beginExtra\n";
    ss<<"endExtra\n";
    // dump string stream to disk;
    file<<ss.rdbuf();
    file.close();
}


int main(int argc, char** argv)
{

    float dt = 0.001f;
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
        dumpToGeo(data.m_pos,data.m_col,i);
    }
*/

    /// GPU VERSION
    ///

    float res = 2048.0f;
    FlockSystem flockSys(n,10.0f,0.1f,dt,1.0f,res);
    flockSys.init();
    std::vector<float3> pos;
    std::vector<float3> col;

    pos.resize(n);
    col.resize(n);

    for(unsigned int i = 0; i < nframes; ++i)
    {
        std::cout << "Timestep="<<dt * float(i+1) << "\n";
        dumpToGeo(pos,col,i);
        flockSys.tick();
        flockSys.exportResult(pos,col);
    }

    /// Test run
    //testing::InitGoogleTest(&argc, argv);
    return 0;//RUN_ALL_TESTS();
}
