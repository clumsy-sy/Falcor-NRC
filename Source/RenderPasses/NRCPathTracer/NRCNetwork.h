#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>

#include "RadianceStructure.slang"
#include "curand.h"



namespace NRC
{
// TODO: Dynamically obtain the resolution and choose the appropriate amount of learning
// full HD {1980, 1080}
const ::uint2 screen_size = {1920, 1080};
// 1980 / 6 = 320; 1080 / 6 = 180 ; 320 * 180 = 57600
const ::uint2 tiling = {6, 6};
// 4 * 16384 = 65536
const uint32_t n_train_batch = 4;
const uint32_t batch_size = 1 << 14; // 16384
// assum 1 pps
const uint32_t pixels_total = screen_size.x * screen_size.y;
// for short path
const uint32_t max_infer_query_size = pixels_total;
// for long train suffix
const uint32_t max_train_query_size = 1 << 16; // 65536
const uint32_t max_train_suffix_length = 5;
// for train sample
const uint32_t max_train_sample_size = max_train_query_size * max_train_suffix_length;
const uint32_t self_query_batch_size = 1 << 16;
// inputBase[float * 16]
const uint32_t input_dim = 16;
// only RGB
const uint32_t output_dim = 3;
// !TODO: Do not use Absolute path
const std::string config_path = "C:/Users/Sy200/Desktop/Falcor-NRC/Source/RenderPasses/NRCPathTracer/config/nrc_default.json";

// tcnn network define

struct NRCNetConfig;
struct NRCMemory;

// on CPU memory
struct NRCCounter
{
    uint32_t train_query_cnt;
    uint32_t train_sample_cnt;
    uint32_t infer_query_cnt;
    uint32_t pad;
};

/**
 * NRC Network use TCNN
 */
class NRCNetwork
{
public:
    NRCNetwork();
    ~NRCNetwork();
    /**
     * @brief NRC network init with a json file
     */
    void InitNRCNetwork();
    /**
     * @brief change learning rate in UI
     */
    float& getLearningRate() { return learning_rate; };
    /**
     * @brief reset network
     * TODO: Completely reset the network!
     */
    void reset();
    /**
     * @brief NRC inference (ALL param is on video memory)
     * @param queries network input array (float 16)
     * @param infer_cnt The number of inference inputs
     * @param pixels Pixels corresponding to input
     * @param output Output to texture
     * @param useRF control if use Reflectance factorization
     */
    __host__ void nrc_inference(inputBase* queries, uint32_t* infer_cnt, ::uint2* pixels, cudaSurfaceObject_t output, bool useRF);
    /**
     * @brief NRC train pass 1. self inference 2. train (ALL param is on video memory)
     * @param self_queries network self reference input array (float 16)
     * @param infer_cnt The number of inference inputs
     * @param train_samples train samples to get {input & target}
     * @param train_samples_cnt The number of train samples
     * @param loss loss in network
     * @param shuffle control if use shuffle
     */
    __host__ void nrc_train(
        inputBase* self_queries,
        uint32_t* self_query_cnt,
        trainSample* train_samples,
        uint32_t* train_samples_cnt,
        float& loss,
        bool shuffle
    );
    /**
     * @brief NRC train pass no self inference 2. train (ALL param is on video memory)
     * @param train_samples train samples to get {input & target}
     * @param train_samples_cnt The number of train samples
     * @param loss loss in network
     * @param shuffle control if use shuffle
     */
    __host__ void nrc_train_simple(trainSample* train_samples, uint32_t* train_samples_cnt, float& loss, bool shuffle);
    /**
     *
     */
    __host__ void copyCountToHost(uint32_t t[4])
    {
        nrc_counter.train_sample_cnt = t[0];
        nrc_counter.infer_query_cnt = t[1];
        nrc_counter.train_query_cnt = t[2];
        nrc_counter.pad = t[3];
        printf(
            "[NRC Counter] : train_sample_cnt = %i, infer_query_cnt = %i, train_query_cnt = %u\n",
            nrc_counter.train_sample_cnt,
            nrc_counter.infer_query_cnt,
            nrc_counter.train_query_cnt
        );
    }
    __host__ void copyCountToHost(std::vector<uint32_t> t)
    {
        nrc_counter.train_sample_cnt = t[0];
        nrc_counter.infer_query_cnt = t[1];
        nrc_counter.train_query_cnt = t[2];
        nrc_counter.pad = t[3];
#ifdef LOG
        printf(
            "[NRC Counter] : train_sample_cnt = %i, infer_query_cnt = %i, train_query_cnt = %u\n",
            nrc_counter.train_sample_cnt,
            nrc_counter.infer_query_cnt,
            nrc_counter.train_query_cnt
        );
#endif
    }

private:
    uint32_t seed = 11086u;
    float learning_rate = 1e-4f;
    uint64_t train_times = 0;

    // cuda
    ::cudaStream_t infer_stream;
    ::cudaStream_t train_stream;
    ::curandGenerator_t rng;

    std::shared_ptr<NRCNetConfig> nrc_network;
    std::shared_ptr<NRCMemory> nrc_memory;
public:
    NRCCounter nrc_counter = {};
};

} // namespace NRC
