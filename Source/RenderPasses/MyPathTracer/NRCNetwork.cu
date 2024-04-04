#include "NRCNetwork.h"
#include "tiny-cuda-nn/common_host.h"
#include "vector_types.h"

#include <cuda.h>
#include <curand.h>

#include <cstdint>
#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

#include <memory>
#include <iostream>

using precision_t = tcnn::network_precision_t;

using GPUMatrix = tcnn::GPUMatrix<float>;

namespace NRC
{
// cuda related {stream, rand}
::cudaStream_t inference_stream;
::cudaStream_t training_stream;
::curandGenerator_t rng;
// network
struct NRCNetConfig
{
    std::shared_ptr<tcnn::Loss<precision_t>> loss = nullptr;
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer = nullptr;
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = nullptr;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer = nullptr;
};

inline NRCNetConfig create_from_config(uint32_t n_input_dims, uint32_t n_output_dims, tcnn::json config)
{
    tcnn::json loss_opts = config.value("loss", tcnn::json::object());
    tcnn::json optimizer_opts = config.value("optimizer", tcnn::json::object());
    tcnn::json network_opts = config.value("network", tcnn::json::object());
    tcnn::json encoding_opts = config.value("encoding", tcnn::json::object());

    std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(loss_opts)};
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizer_opts)};
    auto network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);
    auto trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
    return {loss, optimizer, network, trainer};
}
// network memory
struct NRCMemory
{
    GPUMatrix* training_data = nullptr;
    GPUMatrix* training_target = nullptr;
    GPUMatrix* inference_data = nullptr;
    GPUMatrix* inference_target = nullptr;
    GPUMatrix* training_self_query = nullptr;
    GPUMatrix* training_self_pred = nullptr;
    tcnn::GPUMemory<float>* random_seq = nullptr;
};

struct NRCCounter
{ // pinned memory on device
    uint32_t training_query_count;
    uint32_t training_sample_count;
    uint32_t inference_query_count;
};

NRCMemory* nrc_memory;
std::shared_ptr<NRCNetConfig> nrc_network;
NRCCounter* nrc_counter;
} // namespace NRC

namespace NRC
{

__device__ float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ float3 operator*(float3 a, float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator/(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ float3 safe_div(float3 a, float3 b)
{
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

__device__ void safe_num(float3 &num)
{
    num.x = isinf(num.x) || isnan(num.x) ? 0 : num.x;
    num.y = isinf(num.y) || isnan(num.y) ? 0 : num.y;
    num.z = isinf(num.z) || isnan(num.z) ? 0 : num.z;
}

template<typename T = float>
__global__ void check_nans(uint32_t n_elements, T* data)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements)
        return;
    if (isnan(data[i]) || isinf(data[i]))
    {
        data[i] = (T)0.f;
    }
}

template<typename T>
__device__ void copyInputBase(T* data, const NRC::inputBase* query)
{
    const size_t size = sizeof(NRC::inputBase);
    memcpy(data, query, size);
}

template<uint32_t inputDim, typename T = float>
__global__ void generateBatchSeq(uint32_t n_elements, uint32_t offset, NRC::inputBase* queries, T* data)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset < n_elements)
    {
        uint32_t data_index = i * inputDim, query_index = i + offset;
        copyInputBase(&data[data_index], &queries[query_index]);
    }
}

template<typename T>
__global__ void mapInferenceRadianceToTexture(uint32_t n_elements, T* data, cudaSurfaceObject_t output, uint2* pixels)
{
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > n_elements)
        return;
    uint32_t px = pixels[i].x, py = pixels[i].y;
    uint32_t data_index = i * 3;
    float4 radiance = {data[data_index], data[data_index + 1], data[data_index + 2], 1.0f};
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);
}

template<uint32_t inputDim, typename T = float>
__global__ void mapInferenceRadianceToTextureRR(
    uint32_t n_elements,
    NRC::inputBase* query,
    T* target,
    cudaSurfaceObject_t output,
    uint2* pixels
)
{
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > n_elements)
        return;
    uint32_t px = pixels[i].x, py = pixels[i].y;
    uint32_t idx = i * 3;
    float3 target_rad = {target[idx], target[idx + 1], target[idx + 2]};
    float3 diffuse = query[i].diffuse_refl, specular = query[i].specular_refl;
    float3 rr = target_rad * (diffuse + specular);
    float4 radiance = {rr.x, rr.y, rr.z, 1.0};
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);
}

template<uint32_t inputDim, typename T = float>
__global__ void generateTrainingDataFromSamples(
    uint32_t n_elements,
    uint32_t offset,
    NRC::trainSample* samples,
    NRC::inputBase* self_queries,
    T* self_query_pred,
    T* training_data,
    T* training_target,
    uint32_t* train_sample_cnt,
    uint32_t* self_query_cnt,
    float* random_idx = nullptr
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;
    uint32_t data_index = i * inputDim, sample_index = i + offset;
    if (random_idx)
    {
        sample_index = (1 - random_idx[sample_index]) * (*train_sample_cnt);
    }

    uint32_t pred_index = samples[sample_index].train_idx;

    if (sample_index < *train_sample_cnt)
    {
        float3 radiance = samples[sample_index].radiance;
        uint32_t output_index = i * 3;

        if (pred_index < (*self_query_cnt))
        {
            float3 self_queries_rad = {
                self_query_pred[pred_index * 3], self_query_pred[pred_index * 3 + 1], self_query_pred[pred_index * 3 + 2]
            };
            float3 query_radiance = samples[sample_index].thp * self_queries_rad *
                                    (samples[sample_index].rad.diffuse_refl + samples[sample_index].rad.specular_refl);
            radiance = radiance + query_radiance;
        }

        copyInputBase(&training_data[data_index], &samples[sample_index].rad);
        safe_num(radiance);
        *(float3*)&training_target[output_index] = radiance;
    }
}

template<uint32_t inputDim, typename T = float>
__global__ void generateTrainingDataFromSamples_simple(
    uint32_t n_elements,
    uint32_t offset,
    NRC::trainSample* samples,
    T* training_data,
    T* training_target,
    uint32_t* train_sample_cnt,
    float* random_idx = nullptr
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;
    uint32_t data_index = i * inputDim, sample_index = i + offset;
    if (random_idx)
    {
        sample_index = (1 - random_idx[sample_index]) * (*train_sample_cnt);
    }

    if (sample_index < *train_sample_cnt)
    {
        float3 radiance = samples[sample_index].radiance;
        uint32_t output_index = i * 3;

        copyInputBase(&training_data[data_index], &samples[sample_index].rad);
        safe_num(radiance);
        *(float3*)&training_target[output_index] = radiance;
    }
}

void printInputBase(const inputBase base)
{
    printf("|hit_pos : {%3.2f %3.2f %3.2f}|  ", base.hit_pos.x, base.hit_pos.y, base.hit_pos.z);
    printf("|sca_dir : {%3.2f %3.2f}|  ", base.scatter_dir.x, base.scatter_dir.y);
    printf("|suf_nor : {%3.2f %3.2f}|  ", base.suf_normal.x, base.suf_normal.y);
}

void printSample(const trainSample sample)
{
    printf("[%u]|radiance : {%3.2f %3.2f %3.2f}|  ", sample.train_idx, sample.radiance.x, sample.radiance.y, sample.radiance.z);
    printf("|thp : {%3.2f %3.2f %3.2f}|  --  ", sample.thp.x, sample.thp.y, sample.thp.z);
    printf("|hit_pos : {%3.2f %3.2f %3.2f}|  ", sample.rad.hit_pos.x, sample.rad.hit_pos.y, sample.rad.hit_pos.z);
    printf("|sca_dir : {%3.2f %3.2f}|  ", sample.rad.scatter_dir.x, sample.rad.scatter_dir.y);
    printf("|suf_nor : {%3.2f %3.2f}|  ", sample.rad.suf_normal.x, sample.rad.suf_normal.y);
}

template<typename T>
void showMsg(const T* dataOnDevice, int size)
{
    T* dataOnHost = new T[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        std::cout << dataOnHost[i] << " ";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

uint32_t showMsg_counter(uint32_t* dataOnDevice)
{
    uint32_t* dataOnHost = new uint32_t[1];
    cudaMemcpy(dataOnHost, dataOnDevice, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("%u\n", dataOnHost[0]);
    uint32_t res = dataOnHost[0];
    delete[] dataOnHost;
    return res;
}

void showMsgColor(const float3* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    float3* dataOnHost = new float3[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printf("{%4.2f %4.2f %4.2f}\n", dataOnHost[i].x, dataOnHost[i].y, dataOnHost[i].z);
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showMsgBase(const inputBase* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    inputBase* dataOnHost = new inputBase[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(inputBase), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printInputBase(dataOnHost[i]);
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showMsgSample(const trainSample* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    trainSample* dataOnHost = new trainSample[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(trainSample), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printSample(dataOnHost[i]);
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

NRCNetwork::NRCNetwork()
{
    // infer and train stream
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));
    // rander generator
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 10086ULL);
    curandSetStream(rng, training_stream);
    InitNRCNetwork();
}

NRCNetwork::~NRCNetwork()
{
    cudaError_t result = cudaFreeHost(nrc_counter);
    if (result != cudaSuccess)
    {
        printf("cudaFreeHost failed with error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}

void NRCNetwork::InitNRCNetwork()
{
    cudaError_t result = cudaHostAlloc((void**)&nrc_counter, sizeof(NRCCounter), cudaHostAllocDefault);
    if (result != cudaSuccess)
    {
        printf("cudaHostAlloc failed with error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    // parse json and init network
    std::ifstream f(config_path);
    tcnn::json config = tcnn::json::parse(f);

    nrc_network = std::make_shared<NRCNetConfig>(create_from_config(input_dim, output_dim, config));
    nrc_memory = new NRCMemory();

    learning_rate = nrc_network->optimizer->learning_rate();
    printf(
        "[Learning rate] = %f   [batch_size] = %u   [pixels_total] = %u   [self_query_batch_size] = %u\n",
        learning_rate,
        batch_size,
        pixels_total,
        self_query_batch_size
    );
    nrc_memory->training_data = new GPUMatrix(input_dim, batch_size);
    nrc_memory->training_target = new GPUMatrix(output_dim, batch_size);
    nrc_memory->inference_data = new GPUMatrix(input_dim, pixels_total);
    nrc_memory->inference_target = new GPUMatrix(output_dim, pixels_total);
    nrc_memory->training_self_query = new GPUMatrix(input_dim, self_query_batch_size);
    nrc_memory->training_self_pred = new GPUMatrix(output_dim, self_query_batch_size);

    nrc_memory->random_seq = new tcnn::GPUMemory<float>(n_train_batch * batch_size);
    curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);

    if (nrc_memory->training_data && nrc_memory->training_target && nrc_memory->inference_data && nrc_memory->inference_target &&
        nrc_memory->training_self_query && nrc_memory->training_self_pred && nrc_memory->random_seq)
    {
        printf("Init cuda Success!\n");
    }
    else
    {
        printf("Init cuda Alloc failed!\n");
        exit(-1);
    }
}

void NRCNetwork::reset()
{
    CUDA_CHECK_THROW(cudaStreamSynchronize(training_stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(inference_stream));
    try
    {
        nrc_network->trainer->initialize_params();
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc reset][error]--failed : " << e.what() << std::endl;
        exit(-1);
    }
    printf("[reset network finish] -----------------------------------------------------\n");
}

// query radiance
void NRCNetwork::nrc_inference(
    inputBase* queries,
    uint2* pixels,
    uint32_t* inference_counter,
    uint32_t infer_cnt_on_cpu,
    cudaSurfaceObject_t output,
    bool useRF
)
{
    auto n_elements = infer_cnt_on_cpu;
#ifdef LOG
    printf("[Nrc inference] : inference_query_count = ");
    n_elements = showMsg_counter(inference_counter);
#endif
    // must be 256; Beacuse In "object.h":130 'CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);' !!!
    uint32_t next_batch_size = tcnn::next_multiple(n_elements, 256u);

    if (!n_elements)
        return;
    try
    {
        nrc_memory->inference_target->set_size_unsafe(output_dim, next_batch_size);
        nrc_memory->inference_data->set_size_unsafe(input_dim, next_batch_size);
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--nrc_memory->set_size_unsafe : " << e.what() << std::endl;
        exit(-1);
    }
    try
    {
        tcnn::linear_kernel(generateBatchSeq<input_dim>, 0, inference_stream, n_elements, 0, queries, nrc_memory->inference_data->data());
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--nrc_memory->set_size_unsafe : " << e.what() << std::endl;
        exit(-1);
    }

    try
    {
        nrc_network->network->inference(inference_stream, *nrc_memory->inference_data, *nrc_memory->inference_target);
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--inference :" << e.what() << std::endl;
        exit(-1);
    }

    try
    {
        if (useRF)
        {
            tcnn::linear_kernel(
                mapInferenceRadianceToTextureRR<input_dim>,
                0,
                inference_stream,
                n_elements,
                queries,
                nrc_memory->inference_target->data(),
                output,
                pixels
            );
        }
        else
        {
            tcnn::linear_kernel(
                mapInferenceRadianceToTexture<float>, 0, inference_stream, n_elements, nrc_memory->inference_target->data(), output, pixels
            );
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--mapRadiance : " << e.what() << std::endl;
        exit(-1);
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(inference_stream);
    if (cudaStatus != cudaSuccess)
    {
        printf("CUDA inference error: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
#ifdef LOG
    printf("[Nrc inference %lld] : SUCCESS!\n", train_times);
#endif
}

void NRCNetwork::nrc_train(
    inputBase* self_queries,
    uint32_t* self_query_cnt,
    uint32_t self_query_cnt_on_cpu,
    trainSample* training_samples,
    uint32_t* train_sample_cnt,
    float& loss,
    bool shuffle
)
{
#ifdef LOG
    printf("[net train] : learning_rate = %f ,", train_times, learning_rate);
    printf("train_sample_cnt = ");
    uint32_t sample_cnt = showMsg_counter(train_sample_cnt);
    printf("train_self_query_cnt = ");
    uint32_t self_query_cnt = showMsg_counter(self_query_cnt);
#endif
    nrc_network->optimizer->set_learning_rate(learning_rate);

    // self query
    try
    {
        tcnn::linear_kernel(
            generateBatchSeq<input_dim>, 0, training_stream, self_query_cnt_on_cpu, 0, self_queries, nrc_memory->training_self_query->data()
        );
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc train][error]-- generate self query data : " << e.what() << std::endl;
        exit(-1);
    }

    try
    {
        nrc_network->network->inference(training_stream, *nrc_memory->training_self_query, *nrc_memory->training_self_pred);
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc train][error]-- self query : " << e.what() << std::endl;
        exit(-1);
    }

    curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);
    for (uint32_t i = 0; i < n_train_batch; i++)
    {
        // generate train data and traget
        try
        {
            tcnn::linear_kernel(
                generateTrainingDataFromSamples<input_dim, float>,
                0,
                training_stream,
                batch_size,
                i * batch_size,
                training_samples,
                self_queries,
                nrc_memory->training_self_pred->data(),
                nrc_memory->training_data->data(),
                nrc_memory->training_target->data(),
                train_sample_cnt,
                self_query_cnt,
                shuffle ? nrc_memory->random_seq->data() : nullptr
            );
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--gen train data and target : " << e.what() << std::endl;
            exit(-1);
        }
        // nrc network training
        try
        {
            nrc_network->trainer->training_step(training_stream, *nrc_memory->training_data, *nrc_memory->training_target);
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--Training : " << e.what() << std::endl;
            exit(-1);
        }
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(training_stream);
    if (cudaStatus != cudaSuccess)
    {
        printf("CUDA training error: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
#ifdef LOG
    printf("[net train %lld] : SUCCESS!\n", train_times);
#endif
    train_times++;
}

void NRCNetwork::nrc_train(trainSample* training_samples, uint32_t* train_sample_cnt, float& loss, bool shuffle)
{
#ifdef LOG
    printf("[net simple_train] : learning_rate = %f ,", train_times, learning_rate);
    // if(learning_rate == 0.0) {
    //     printf("learning_rate == 0!!!\n");
    //     exit(-1);
    // }
    printf("train_sample_cnt = ");
    uint32_t sample_cnt = showMsg_counter(train_sample_cnt);
#endif
    nrc_network->optimizer->set_learning_rate(learning_rate);

    curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);
    for (uint32_t i = 0; i < n_train_batch; i++)
    {
        // generate train data and traget
        try
        {
            tcnn::linear_kernel(
                generateTrainingDataFromSamples_simple<input_dim, float>,
                0,
                training_stream,
                batch_size,
                i * batch_size,
                training_samples,
                nrc_memory->training_data->data(),
                nrc_memory->training_target->data(),
                train_sample_cnt,
                shuffle ? nrc_memory->random_seq->data() : nullptr
            );
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--gen train data and target : " << e.what() << std::endl;
            exit(-1);
        }
        // nrc network training
        try
        {
            nrc_network->trainer->training_step(training_stream, *nrc_memory->training_data, *nrc_memory->training_target);
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--Training : " << e.what() << std::endl;
            exit(-1);
        }
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(training_stream);
    if (cudaStatus != cudaSuccess)
    {
        printf("CUDA training error: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
#ifdef LOG
    printf("[net simple_train %lld] : SUCCESS!\n", train_times);
#endif
    train_times++;
}

} // namespace NRC
