#ifndef GEOEXPORTER_H_
#define GEOEXPORTER_H_
#include <vector>
#include <glm/vec3.hpp>
#include "helper_math.h"

namespace GeoExporter
{

    void dumpToGeo(const std::vector<glm::vec3> &points,
                const std::vector<glm::vec3> &colour,
                const unsigned int cnt);


    void dumpToGeo(const std::vector<float3> &points,
                const std::vector<float3> &colour,
                const unsigned int cnt);

}

#endif // GEOEXPORTER_H_
