#include "NRCInterface.h"
#include <cstdint>
#include "Core/Object.h"
#include "vector_types.h"

// #ifndef LOG
// #define LOG
// #endif

namespace MININRC
{
NRCInterface::NRCInterface(Falcor::ref<Falcor::Device> pDevice)
{
    // Initialize CUDA device
    if (!pDevice.get()->initCudaDevice())
        FALCOR_THROW("Failed to initialize CUDA device.");

    Falcor::logInfo("NRCInterface::working directory: " + std::filesystem::current_path().string());
    Falcor::logInfo("NRCInferface::creating and initializing network");

    nrc_network_ref = std::make_shared<NRCNetwork>();
    pDevice = pDevice;
}

NRCInterface::~NRCInterface()
{
    // if (mCudaResources.screen_query != nullptr)
    // {
    //     free(mCudaResources.screen_query);
    // }
    // if (mCudaResources.train_query != nullptr)
    // {
    //     free(mCudaResources.train_query);
    // }
    // if (mCudaResources.train_sample != nullptr)
    // {
    //     free(mCudaResources.train_sample);
    // }
    // if (mCudaResources.infer_query_pixel != nullptr)
    // {
    //     free(mCudaResources.infer_query_pixel);
    // }
    // if (mCudaResources.counter_buffer_ptr != nullptr)
    // {
    //     free(mCudaResources.counter_buffer_ptr);
    // }
    // if (mCudaResources.train_query_cnt != nullptr)
    // {
    //     free(mCudaResources.train_query_cnt);
    // }
    // if (mCudaResources.train_sample_cnt != nullptr)
    // {
    //     free(mCudaResources.train_sample_cnt);
    // }
    nrc_network_ref->reset();
    pDevice.reset();
}

void NRCInterface::train(uint32_t train_cnt, uint32_t self_queriy_cnt, bool shuffle)
{
    float loss;
    if (mCudaResources.train_query == nullptr || mCudaResources.train_query_cnt == nullptr ||
        mCudaResources.train_sample == nullptr || mCudaResources.train_sample_cnt == nullptr)
    {
        std::cerr << "[ERROR] : NRCInterface::trainFrame -> train resources is Empty!\n";
        exit(-1);
    }
    nrc_network_ref->nrc_train(
        mCudaResources.train_query,
        mCudaResources.train_query_cnt,
        mCudaResources.train_sample,
        mCudaResources.train_sample_cnt,
        loss,
        shuffle
    );
}

void NRCInterface::trainSimple(uint32_t train_cnt, bool shuffle)
{
    float loss = nrc_network_ref->getLearningRate();
    if (mCudaResources.train_sample == nullptr || mCudaResources.train_sample_cnt == nullptr)
    {
        std::cerr << "[ERROR] : NRCInterface::trainFrame -> train resources is Empty!\n";
        exit(-1);
    }

    nrc_network_ref->nrc_train_simple(mCudaResources.train_sample, mCudaResources.train_sample_cnt, loss, shuffle);
}

void NRCInterface::inference(uint32_t infer_cnt, bool useRF)
{
    nrc_network_ref->nrc_inference(
        mCudaResources.screen_query,
        mCudaResources.infer_cnt,
        mCudaResources.infer_query_pixel,
        mCudaResources.screen_result,
        useRF
    );
}

void NRCInterface::log()
{
    std::stringstream ss;
    Falcor::logInfo(ss.str());
}

void NRCInterface::reset()
{
    printf("[clear]--------------------------------------------------------------------------\n");
    nrc_network_ref->reset();
}

void NRCInterface::mapResources(
    Falcor::ref<Falcor::Buffer> pScreenQueryBuffer,
    Falcor::ref<Falcor::Buffer> pTrainingQueryBuffer,
    Falcor::ref<Falcor::Buffer> pTrainingSampleBuffer,
    Falcor::ref<Falcor::Buffer> pSharedCounterBuffer,
    Falcor::ref<Falcor::Buffer> pInferenceRadiancePixel,
    Falcor::ref<Falcor::Texture> pScreenResultTexture
)
{
    if (pScreenResultTexture.get() == nullptr)
    {
        Falcor::logWarning("pScreenResultTexture is Empty!\n");
        exit(-1);
    }
    mCudaResources.screen_result = Falcor::cuda_utils::mapTextureToSurface(pScreenResultTexture, cudaArrayColorAttachment);
    mCudaResources.screen_query = (MININRC::inputBase*)pScreenQueryBuffer.get()->getCudaMemory()->getMappedData();
    mCudaResources.train_query = (MININRC::inputBase*)pTrainingQueryBuffer.get()->getCudaMemory()->getMappedData();
    mCudaResources.train_sample = (MININRC::trainSample*)pTrainingSampleBuffer.get()->getCudaMemory()->getMappedData();
    mCudaResources.infer_query_pixel = (uint2*)pInferenceRadiancePixel.get()->getCudaMemory()->getMappedData();
    uint32_t* counterBuffer = (uint32_t*)pSharedCounterBuffer.get()->getCudaMemory()->getMappedData();
    mCudaResources.counter_buffer_ptr = counterBuffer;
    mCudaResources.train_sample_cnt = &counterBuffer[0];
    mCudaResources.infer_cnt = &counterBuffer[1];
    mCudaResources.train_query_cnt = &counterBuffer[2];
}

} // namespace NRC
