#pragma once
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "RadianceStructure.slang"

// import RadianceStructure;
import Scene.Raytracing;
import Scene.ShadingData;
import Utils.Math.MathHelpers;

RWStructuredBuffer<MININRC::inputBase> gTrainingRadianceQuery;
RWStructuredBuffer<MININRC::trainSample> gTrainingtrainSample;
RWStructuredBuffer<MININRC::inputBase> gInferenceRadianceQuery;
RWStructuredBuffer<MININRC::inputBase> gTmpPathRecord;
RWStructuredBuffer<uint2> gInferenceRadiancePixel;

RWTexture2D<float4> gScreenQueryFactor;
RWTexture2D<float4> gScreenQueryBias;

Buffer<uint> counter;

namespace MININRC
{
    enum NRCPathType
    {
        InferPath = 0,
        TrainPath = 1
    };

    struct PathVertexRecord
    {
        inputBase record;
        float3 thp; // current path throughput (not including BxDF, pdf of this vertex)
        uint path_id;
        float3 L; // current path contribution excluding scattering radiance from this vertex
        uint pad;

        uint2 getPixel() {
            return uint2(path_id & 0x00000FFF, (path_id >> 12) & 0x00000FFF);
        }
    };

    inputBase generateQuery(ShadingData sd)
    {
        inputBase base = {};
        base.hit_pos = sd.posW;
        base.scatter_dir = world_to_latlong_map(sd.V);
        float3 faceN = sd.frontFacing ? sd.faceN : -sd.faceN;
        base.suf_normal = world_to_latlong_map(faceN);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        base.diffuse_refl = mi.getProperties(sd).diffuseReflectionAlbedo;
        base.specular_refl = mi.getProperties(sd).specularReflectionAlbedo;
        base.suf_roughness = 1 - exp(-mi.getProperties(sd).roughness);
        return base;
    }

    PathVertexRecord recordNode(ShadingData sd, float3 radiance, float3 thp)
    {
        inputBase base = generateQuery(sd);
        PathVertexRecord record = { base, thp, 0, radiance, 0 };
        return record;
    }
    PathVertexRecord recordNode(ShadingData sd, uint2 pixel, float3 radiance, float3 thp)
    {
        inputBase base = generateQuery(sd);
        uint path_id = pixel.x | (pixel.y << 12);
        PathVertexRecord record = { base, thp, path_id, radiance, 0 };
        return record;
    }

    uint addPathRecord(ShadingData sd, float3 thp) {
        uint idx = gTmpPathRecord.IncrementCounter();
        float2 thpmap = world_to_latlong_map(thp);
        inputBase base = generateQuery(sd);
        base.pad0 = thpmap.x;
        base.pad0 = thpmap.y;
        gTmpPathRecord[idx] = base;
        return idx;
    }

    PathVertexRecord reconstructPathRecord(uint idx, float3 radiance, uint2 pixel) {
        inputBase base = gTmpPathRecord[idx];
        float2 thpmap = { base.pad0, base.pad1 };
        float3 thp = latlong_map_to_world(thpmap);
        base.pad0 = 0.0;
        base.pad1 = 0.0;
        uint path_id = pixel.x | (pixel.y << 12);
        PathVertexRecord record = { base, thp, path_id, radiance, 0 };
        return record;
    }
    PathVertexRecord reconstructPathRecord(uint idx, float3 radiance, float3 thp, uint2 pixel) {
        inputBase base = gTmpPathRecord[idx];
        uint path_id = pixel.x | (pixel.y << 12);
        PathVertexRecord record = { base, thp, path_id, radiance, 0 };
        return record;
    }

    void writeInferQuery(uint2 pixel, ShadingData sd, float3 bias, float3 thp)
    {
        uint index = gInferenceRadianceQuery.IncrementCounter();
        gInferenceRadianceQuery[index] = generateQuery(sd);
        gInferenceRadiancePixel[index] = pixel;
        gScreenQueryFactor[pixel] = float4(thp, 1.0f);
        gScreenQueryBias[pixel] = float4(bias, 1.0f);
    }

    void writeInferQuery(uint2 pixel, inputBase base, float3 thp, float3 bias)
    {
        uint index = gInferenceRadianceQuery.IncrementCounter();
        gInferenceRadianceQuery[index] = base;
        gInferenceRadiancePixel[index] = pixel;
        gScreenQueryFactor[pixel] = float4(thp, 1.0f);
        gScreenQueryBias[pixel] = float4(bias, 1.0f);
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
        sample.input = query;
        sample.train_idx = num;
        sample.L = radiance;
        sample.thp = thp;
        gTrainingtrainSample[idx] = sample;
    }

    float3 safe_div(float3 a, float3 b)
    {
        float3 res = a / b;
        res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
        res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
        res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
        return res;
    }

}
