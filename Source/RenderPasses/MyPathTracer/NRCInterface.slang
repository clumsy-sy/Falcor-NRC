#pragma once
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "RadianceStructure.slang"

// import RadianceStructure;
import Scene.Raytracing;
import Scene.ShadingData;
import Utils.Math.MathHelpers;

#ifndef NRC_MAX_TRAINING_BOUNCES
#define NRC_MAX_TRAINING_BOUNCES 4
#endif
#ifndef NRC_MAX_INFERENCE_BOUNCES
#define NRC_MAX_INFERENCE_BOUNCES 2
#endif

RWStructuredBuffer<NRC::inputBase> gTrainingRadianceQuery;
RWStructuredBuffer<NRC::trainSample> gTrainingtrainSample;
RWStructuredBuffer<NRC::inputBase> gInferenceRadianceQuery;
RWStructuredBuffer<uint2> gInferenceRadiancePixel;

RWTexture2D<float4> gScreenQueryFactor;
RWTexture2D<float4> gScreenQueryBias;

Buffer<uint> counter;

namespace NRC
{
    enum NRCPathType
    {
        InferencePath = 0,
        TrainingPath = 1,
        TrainingPathRR = 2
    };

    struct PathVertexRecord
    {
        inputBase query;
        float3 thp; // current path throughput (not including BxDF, pdf of this vertex)
        float3 L;   // current path contribution excluding scattering radiance from this vertex
    };

    float3 safe_div(float3 a, float3 b)
    {
        float3 res = a / b;
        res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
        res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
        res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
        return res;
    }

    inputBase generateQuery(ShadingData sd)
    {
        inputBase query = {};
        query.hit_pos = sd.posW;
        query.scatter_dir = world_to_latlong_map(sd.V);
        float3 faceN = sd.frontFacing ? sd.faceN : -sd.faceN;
        query.suf_normal = world_to_latlong_map(faceN);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        query.diffuse_refl = mi.getProperties(sd).diffuseReflectionAlbedo;
        query.specular_refl = mi.getProperties(sd).specularReflectionAlbedo;
        query.suf_roughness = 1 - exp(-mi.getProperties(sd).roughness);
        return query;
    }

    PathVertexRecord recordNode(ShadingData sd, float3 radiance, float3 thp)
    {
        inputBase query = generateQuery(sd);
        PathVertexRecord record = { query, thp, radiance };
        return record;
    }

    void writeScreenInferenceQuery(uint2 pixel, ShadingData sd, float3 bias, float3 thp)
    {
        uint index = gInferenceRadianceQuery.IncrementCounter();
        gInferenceRadianceQuery[index] = generateQuery(sd);
        gInferenceRadiancePixel[index] = pixel;
        gScreenQueryFactor[pixel] = float4(thp, 1);
        gScreenQueryBias[pixel] = float4(bias, 1);
    }

    uint addTrainingQuery(ShadingData sd)
    {
        uint idx = gTrainingRadianceQuery.IncrementCounter();
        inputBase query = generateQuery(sd);
        gTrainingRadianceQuery[idx] = query;
        return idx;
    }

    void addTrainingSample(float3 radiance, float3 thp, inputBase query, int num)
    {
        uint idx = gTrainingtrainSample.IncrementCounter();
        trainSample sample = {};
        sample.rad = query;
        sample.train_idx = num;
        sample.radiance = radiance;
        sample.thp = thp;
        gTrainingtrainSample[idx] = sample;
    }

}
