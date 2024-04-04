#pragma once
#ifndef RADIANCESTRUCTURE
#define RADIANCESTRUCTURE
// #include "Utils/HostDeviceShared.slangh"

namespace NRC
{
    /**
       Parameter                size           with Encoding
       position                   3                 3 * 12 = 36
       Scatter dir                2                 2 * 4 = 8
       Surface normal             2                 2 * 4 = 8
       Surface roughness          1                 1 * 4 = 4
       diffuse reflection         3                   3   = 3
       specular reflection        3                   3   = 3
       pad0                       1                   1
       pad1                       1                   1
       TOTAL                      16                  64
    */
    struct inputBase
    {
        float3 hit_pos;
        float suf_roughness;
        float2 scatter_dir;
        float2 suf_normal;
        float3 diffuse_refl;
        float pad0;
        float3 specular_refl;
        float pad1;
    };

    struct trainSample
    {
        inputBase rad;
        uint32_t train_idx;
        float3 radiance;
        float3 thp;
        float pad0;
    };

    enum class NRCMode
    {
        Result = 1,
        Radiance = 2,
        Factor = 3,
        Bias = 4,
        Reflectance = 5
    };

}

#endif
