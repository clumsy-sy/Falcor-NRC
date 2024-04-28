#include <algorithm>
#include <cstdint>
#include <memory>
#include "NRCNetwork.h"

// tcnn
#include <cstdint>
#include <json/json.hpp>
#include <vector>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

#include "NRCHelper.cuh"
namespace MININRC
{

using precision_t = tcnn::network_precision_t;

struct NRCNetConfig
{
    std::shared_ptr<tcnn::Loss<precision_t>> loss = nullptr;
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer = nullptr;
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = nullptr;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer = nullptr;
};

inline NRCNetConfig create_from_config(uint32_t n_input_dims, uint32_t n_output_dims, nlohmann::json config)
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
    // TODO: Use smart pointers
    tcnn::GPUMatrix<float>* train_data = nullptr;
    tcnn::GPUMatrix<float>* train_target = nullptr;
    tcnn::GPUMatrix<float>* infer_data = nullptr;
    tcnn::GPUMatrix<float>* infer_result = nullptr;
    tcnn::GPUMatrix<float>* train_self_query = nullptr;
    tcnn::GPUMatrix<float>* train_self_result = nullptr;
    tcnn::GPUMemory<uint32_t>* random_seq = nullptr;
};

// cuda function is gloable, so carefully, do not redefine.
// __device__ float3 operator+(float3 a, float3 b);
// __device__ float3 operator*(float3 a, float3 b);
// __device__ float3 operator/(float3 a, float3 b);
// __device__ float3 safe_div(float3 a, float3 b);
// __device__ void safe_num(float3& num);
// void printInputBase(const inputBase base);
// void printSample(const trainSample sample);
// uint32_t showMsg_counter(uint32_t* dataOnDevice);
// void showMsgColor(const float3* dataOnDevice, int size, int maxsize = 8);
// void showMsgBase(const inputBase* dataOnDevice, int size, int maxsize = 8);
// void showMsgSample(const trainSample* dataOnDevice, int size, int maxsize = 8);

/*
 * @brief copy network input struct to T*
 */
template<typename T>
__device__ void copyInputBase(T* data, const MININRC::inputBase* query)
{
    const size_t size = sizeof(MININRC::inputBase);
    memcpy(data, query, size);
}
/*
 * @brief copy network input struct to T*
 */
template<uint32_t inputDim, typename T = float>
__global__ void genBatchSeq(uint32_t n_elements, uint32_t offset, MININRC::inputBase* queries, T* data)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset < n_elements)
    {
        uint32_t data_index = i * inputDim, query_index = i + offset;
        copyInputBase(&data[data_index], &queries[query_index]);
    }
}
/*
 * @brief map network inference result to a Surface
 */
template<typename T>
__global__ void mapInferenceResultToSurface(uint32_t n_elements, T* data, cudaSurfaceObject_t output, uint2* pixels)
{
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > n_elements)
        return;
    uint32_t px = pixels[i].x, py = pixels[i].y;
    uint32_t data_index = i * 3;
    float4 radiance = {data[data_index], data[data_index + 1], data[data_index + 2], 1.0f};
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);
}
/*
 * @brief map network inference result to a Surface with Reflectance factorization
 */
template<uint32_t inputDim, typename T = float>
__global__ void mapInferenceResultToSurfaceWithRF(
    uint32_t n_elements,
    MININRC::inputBase* query,
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
    // Reflectance factorization
    float3 diffuse = query[i].diffuse_refl, specular = query[i].specular_refl;
    float3 rr = target_rad * (diffuse + specular);
    float4 radiance = {rr.x, rr.y, rr.z, 1.0};
    surf2Dwrite(radiance, output, (int)sizeof(float4) * px, py);
}
/*
 * @brief generate train data {input & target} this function has self query
 */
template<uint32_t inputDim, typename T = float>
__global__ void genTrainDataFromSamples(
    uint32_t n_elements,
    uint32_t offset,
    MININRC::trainSample* samples,
    uint32_t* train_sample_cnt,
    MININRC::inputBase* self_queries,
    uint32_t* self_query_cnt,
    T* self_query_result,
    T* train_data,
    T* train_target,
    // float* random_idx = nullptr
    uint32_t* random_idx = nullptr
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;
    uint32_t data_index = i * inputDim, sample_index = i + offset;
    if (random_idx)
    {
        // sample_index = (1 - random_idx[sample_index]) * (*train_sample_cnt);
        sample_index = random_idx[sample_index];
    }

    uint32_t pred_index = samples[sample_index].train_idx;

    if (sample_index < *train_sample_cnt)
    {
        float3 radiance = samples[sample_index].L;
        uint32_t output_index = i * 3;

        if (pred_index < (*self_query_cnt))
        {
            float3 self_queries_rad = {
                self_query_result[pred_index * 3], self_query_result[pred_index * 3 + 1], self_query_result[pred_index * 3 + 2]
            };
            float3 query_radiance = samples[sample_index].thp * self_queries_rad *
                                    (samples[sample_index].input.diffuse_refl + samples[sample_index].input.specular_refl);
            radiance = radiance + query_radiance;
        }

        copyInputBase(&train_data[data_index], &samples[sample_index].input);
        safe_num(radiance);
        *(float3*)&train_target[output_index] = radiance;
    }
}
/*
 * @brief generate train data {input & target} no self query
 */
template<uint32_t inputDim, typename T = float>
__global__ void genTrainDataFromSamplesSimple(
    uint32_t n_elements,
    uint32_t offset,
    MININRC::trainSample* samples,
    uint32_t* train_sample_cnt,
    T* train_data,
    T* train_target,
    // float* random_idx = nullptr
    uint32_t* random_idx = nullptr
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;
    uint32_t data_index = i * inputDim, sample_index = i + offset;
    if (random_idx)
    {
        // sample_index = (1 - random_idx[sample_index]) * (*train_sample_cnt);
        sample_index = random_idx[sample_index];
    }

    if (sample_index < *train_sample_cnt)
    {
        float3 radiance = samples[sample_index].L;
        uint32_t output_index = i * 3;

        copyInputBase(&train_data[data_index], &samples[sample_index].input);
        safe_num(radiance);
        *(float3*)&train_target[output_index] = radiance;
    }
}

NRCNetwork::NRCNetwork()
{
    // infer and train stream
    CUDA_CHECK_THROW(cudaStreamCreate(&infer_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&train_stream));
    // rander generator
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    curandSetStream(rng, train_stream);
    InitNRCNetwork();
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

    learning_rate = nrc_network->optimizer->learning_rate();
    printf(
        "[Learning rate] = %f   [batch_size] = %u   [pixels_total] = %u   [self_query_batch_size] = %u\n",
        learning_rate,
        batch_size,
        pixels_total,
        self_query_batch_size
    );
    nrc_memory = std::make_shared<NRCMemory>();
    nrc_memory->train_data = new tcnn::GPUMatrix<float>(input_dim, batch_size);
    nrc_memory->train_target = new tcnn::GPUMatrix<float>(output_dim, batch_size);
    nrc_memory->infer_data = new tcnn::GPUMatrix<float>(input_dim, pixels_total);
    nrc_memory->infer_result = new tcnn::GPUMatrix<float>(output_dim, pixels_total);
    nrc_memory->train_self_query = new tcnn::GPUMatrix<float>(input_dim, self_query_batch_size);
    nrc_memory->train_self_result = new tcnn::GPUMatrix<float>(output_dim, self_query_batch_size);
    nrc_memory->random_seq = new tcnn::GPUMemory<uint32_t>(n_train_batch * batch_size);
    // curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);

    // test shuffle
    std::vector<uint32_t> rand_seq(65536);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::shuffle(rand_seq.begin(), rand_seq.end(), rng);
    cudaMemcpy(nrc_memory->random_seq->data(), rand_seq.data(), sizeof(uint32_t) * 65536, cudaMemcpyHostToDevice);

    uint32_t* arr = new uint32_t[65536];
    cudaMemcpy(arr, nrc_memory->random_seq->data(), sizeof(uint32_t) * 65536, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 100; i ++) {
        std::cout << arr[i] << " ";
    }

    //

    if (nrc_memory->train_data && nrc_memory->train_target && nrc_memory->infer_data && nrc_memory->infer_result &&
        nrc_memory->train_self_query && nrc_memory->train_self_result && nrc_memory->random_seq)
    {
        printf("Alloc cuda memory Success!\n");
    }
    else
    {
        printf("Alloc cuda memory Failed!\n");
        exit(-1);
    }
}

NRCNetwork::~NRCNetwork()
{
    // nrc_memory->train_data.reset();
    // nrc_memory->train_target.reset();
    // nrc_memory->infer_data.reset();
    // nrc_memory->infer_result.reset();
    // nrc_memory->train_self_query.reset();
    // nrc_memory->train_self_result.reset();
    // nrc_memory->random_seq.reset();
    // cudaError_t result = cudaFreeHost(nrc_counter);
    // if (result != cudaSuccess)
    // {
    //     printf("cudaFreeHost failed with error: %s\n", cudaGetErrorString(result));
    //     exit(-1);
    // }
}

void NRCNetwork::reset()
{
    CUDA_CHECK_THROW(cudaStreamSynchronize(train_stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(infer_stream));
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

void NRCNetwork::nrc_inference(inputBase* queries, uint32_t* infer_cnt, ::uint2* pixels, cudaSurfaceObject_t output, bool useRF)
{
#ifdef LOG

    // printf(
    //     "[Nrc infer] : inference_query_count = %d , train_query_count = %d, train_sample = %d",
    //     nrc_counter.infer_query_cnt,
    //     nrc_counter.train_query_cnt,
    //     nrc_counter.train_sample_cnt
    // );
#endif
    // must be 256; Beacuse In "object.h":130 'CHECK_THROW(input.n() % BATCH_SIZE_GRANULARITY == 0);' !!!
    uint32_t next_batch_size = tcnn::next_multiple(nrc_counter.infer_query_cnt, 256u);

    try
    {
        nrc_memory->infer_data->set_size_unsafe(input_dim, next_batch_size);
        nrc_memory->infer_result->set_size_unsafe(output_dim, next_batch_size);
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--nrc_memory->set_size_unsafe : " << e.what() << std::endl;
        exit(-1);
    }
    try
    {
        tcnn::linear_kernel(
            genBatchSeq<input_dim>, 0, infer_stream, nrc_counter.infer_query_cnt, 0, queries, nrc_memory->infer_data->data()
        );
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--nrc_memory->set_size_unsafe : " << e.what() << std::endl;
        exit(-1);
    }

    try
    {
        nrc_network->network->inference(infer_stream, *nrc_memory->infer_data, *nrc_memory->infer_result);
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
                mapInferenceResultToSurfaceWithRF<input_dim>,
                0,
                infer_stream,
                nrc_counter.infer_query_cnt,
                queries,
                nrc_memory->infer_result->data(),
                output,
                pixels
            );
        }
        else
        {
            tcnn::linear_kernel(
                mapInferenceResultToSurface<float>,
                0,
                infer_stream,
                nrc_counter.infer_query_cnt,
                nrc_memory->infer_result->data(),
                output,
                pixels
            );
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc inference][error]--mapRadiance : " << e.what() << std::endl;
        exit(-1);
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(infer_stream);
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
    trainSample* train_samples,
    uint32_t* train_samples_cnt,
    float& loss,
    bool shuffle
)
{
#ifdef LOG
    printf("[net train] : learning_rate = %f ,", learning_rate);
    printf("train_samples_cnt = ");
    uint32_t sample_cnt = showMsg_counter(train_samples_cnt);
    printf("train_self_query_cnt = ");
    uint32_t self_querys_cnt = showMsg_counter(self_query_cnt);
#endif
    nrc_network->optimizer->set_learning_rate(learning_rate);

    // self query
    try
    {
        tcnn::linear_kernel(
            genBatchSeq<input_dim>, 0, train_stream, nrc_counter.infer_query_cnt, 0, self_queries, nrc_memory->train_self_query->data()
        );
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc train][error]-- generate self query data : " << e.what() << std::endl;
        exit(-1);
    }

    try
    {
        nrc_network->network->inference(train_stream, *nrc_memory->train_self_query, *nrc_memory->train_self_result);
    }
    catch (const std::exception& e)
    {
        std::cout << "[Nrc train][error]-- self query : " << e.what() << std::endl;
        exit(-1);
    }

    // curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);
    for (uint32_t i = 0; i < n_train_batch; i++)
    {
        // generate train data and traget
        try
        {
            tcnn::linear_kernel(
                genTrainDataFromSamples<input_dim, float>,
                0,
                train_stream,
                batch_size,
                i * batch_size,
                train_samples,
                train_samples_cnt,
                self_queries,
                self_query_cnt,
                nrc_memory->train_self_result->data(),
                nrc_memory->train_data->data(),
                nrc_memory->train_target->data(),
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
            nrc_network->trainer->training_step(train_stream, *nrc_memory->train_data, *nrc_memory->train_target);
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--Training : " << e.what() << std::endl;
            exit(-1);
        }
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(train_stream);
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

void NRCNetwork::nrc_train_simple(trainSample* train_samples, uint32_t* train_samples_cnt, float& learning_rate, bool shuffle)
{
    if(learning_rate <= 0.0 + 1e-8) {
        printf("learning_rate == 0!!!\n");
        exit(-1);
    }
#ifdef LOG
    printf("[net simple_train] : learning_rate = %f ,", train_times, learning_rate);
    printf("train_sample_cnt = ");
    uint32_t sample_cnt = showMsg_counter(train_samples_cnt);
#endif
    nrc_network->optimizer->set_learning_rate(learning_rate);

    // curandGenerateUniform(rng, nrc_memory->random_seq->data(), n_train_batch * batch_size);

    if(nrc_counter.train_sample_cnt < (1>>14)) {
        printf("%u", nrc_counter.train_sample_cnt);
        exit(-1);
    }

    for (uint32_t i = 0; i < n_train_batch; i++)
    {
        // generate train data and traget
        try
        {
            tcnn::linear_kernel(
                genTrainDataFromSamplesSimple<input_dim, float>,
                0,
                train_stream,
                batch_size,
                i * batch_size,
                train_samples,
                train_samples_cnt,
                nrc_memory->train_data->data(),
                nrc_memory->train_target->data(),
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
            nrc_network->trainer->training_step(train_stream, *nrc_memory->train_data, *nrc_memory->train_target);
        }
        catch (const std::exception& e)
        {
            std::cout << "[Nrc simple_train][error]--Training : " << e.what() << std::endl;
            exit(-1);
        }
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(train_stream);
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

} // namespace MININRC
