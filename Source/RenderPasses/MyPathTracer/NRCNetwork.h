#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <fstream>

#include "RadianceStructure.slang"

namespace NRC
{

// full HD {1980, 1080}
const ::uint2 screen_size = {1920, 1080};
// 1980 / 6 = 320; 1080 / 6 = 180 ;320 * 180 = 57600
const ::uint2 tiling = {6, 6};
const uint32_t n_train_batch = 4;
const uint32_t batch_size = 1 << 14; // 16384
// assum 1 pps
const uint32_t pixels_total = screen_size.x * screen_size.y;
const uint32_t max_inference_query_size = pixels_total;
const uint32_t max_training_query_size = 1 << 16; // ~57,600

const uint32_t max_training_sample_size = pixels_total / tiling.x / tiling.y * 5;
const uint32_t self_query_batch_size = pixels_total / tiling.x / tiling.y; // ~ 57600
// pos dir normal roughness diffuse specular pad0 pad1
const uint32_t input_dim = 16;
// RGB
const uint32_t output_dim = 3;
// !TODO: Do not use Absolute path
const std::string config_path = "C:/Users/Sy200/Desktop/Falcor/Source/RenderPasses/MyPathTracer/config/nrc_default.json";

/**
 * NRC Network use TCNN
 */
class NRCNetwork
{
public:
    NRCNetwork();
    ~NRCNetwork();

    void InitNRCNetwork();
    void reset();
    float& getLearningRate() { return learning_rate; };

    __host__ void nrc_inference(
        inputBase* queries,
        ::uint2* pixels,
        uint32_t* inference_counter,
        uint32_t infer_cnt_in_cpu,
        cudaSurfaceObject_t output,
        bool useRF
    );
    __host__ void nrc_train(
        inputBase* self_queries,
        uint32_t* self_query_cnt,
        uint32_t self_query_cnt_on_cpu,
        trainSample* training_samples,
        uint32_t* training_sample_counter,
        float& loss,
        bool shuffle
    );
    __host__ void nrc_train(trainSample* training_samples, uint32_t* training_sample_counter, float& loss, bool shuffle);

private:
    uint32_t seed = 11086u;
    float learning_rate = 1e-4f;
    uint64_t train_times = 0;
};

} // namespace NRC
