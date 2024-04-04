#include "NRCInterface.h"
#include "Core/Object.h"
#include "vector_types.h"

// #ifndef LOG
// #define LOG
// #endif

namespace NRC
{
NRCInterface::NRCInterface(Falcor::ref<Falcor::Device> pDevice)
{
    // Initialize CUDA device
    if (!pDevice.get()->initCudaDevice())
        FALCOR_THROW("Failed to initialize CUDA device.");

    Falcor::logInfo("NRCInterface::working directory: " + std::filesystem::current_path().string());
    Falcor::logInfo("NRCInferface::creating and initializing network");

    nrc_Net_ref = std::make_shared<NRCNetwork>();
    pDevice = pDevice;
}

NRCInterface::~NRCInterface()
{
    // if (cudaResources.screenQuery != nullptr)
    // {
    //     free(cudaResources.screenQuery);
    // }
    // if (cudaResources.trainingQuery != nullptr)
    // {
    //     free(cudaResources.trainingQuery);
    // }
    // if (cudaResources.trainingSample != nullptr)
    // {
    //     free(cudaResources.trainingSample);
    // }
    // if (cudaResources.inferenceQueryPixel != nullptr)
    // {
    //     free(cudaResources.inferenceQueryPixel);
    // }
    // if (cudaResources.counterBufferPtr != nullptr)
    // {
    //     free(cudaResources.counterBufferPtr);
    // }
    // if (cudaResources.trainingQueryCounter != nullptr)
    // {
    //     free(cudaResources.trainingQueryCounter);
    // }
    // if (cudaResources.trainingSampleCounter != nullptr)
    // {
    //     free(cudaResources.trainingSampleCounter);
    // }
    nrc_Net_ref->reset();
    pDevice.reset();
}

void NRCInterface::trainFrame(uint32_t train_cnt, uint32_t self_queriy_cnt, bool shuffle)
{
    float loss;
    if (cudaResources.trainingQuery == nullptr || cudaResources.trainingQueryCounter == nullptr ||
        cudaResources.trainingSample == nullptr || cudaResources.trainingSampleCounter == nullptr)
    {
        std::cerr << "[ERROR] : NRCInterface::trainFrame -> train resources is Empty!\n";
        exit(-1);
    }
    nrc_Net_ref->nrc_train(
        cudaResources.trainingQuery,
        cudaResources.trainingQueryCounter,
        self_queriy_cnt,
        cudaResources.trainingSample,
        cudaResources.trainingSampleCounter,
        loss,
        shuffle
    );

    mStats.n_frames++;
    mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;
}

void NRCInterface::trainSampleFrame(uint32_t train_cnt, bool shuffle)
{
    float loss;
    if (cudaResources.trainingSample == nullptr || cudaResources.trainingSampleCounter == nullptr)
    {
        std::cerr << "[ERROR] : NRCInterface::trainFrame -> train resources is Empty!\n";
        exit(-1);
    }

    nrc_Net_ref->nrc_train(cudaResources.trainingSample, cudaResources.trainingSampleCounter, loss, shuffle);
    mStats.n_frames++;
    mStats.training_loss_avg = mStats.ema_factor * mStats.training_loss_avg + (1 - mStats.ema_factor) * loss;
}

void NRCInterface::inferenceFrame(uint32_t infer_cnt, bool useRF)
{
    nrc_Net_ref->nrc_inference(
        cudaResources.screenQuery,
        cudaResources.inferenceQueryPixel,
        cudaResources.inferenceCounter,
        infer_cnt,
        cudaResources.screenResult,
        useRF
    );
}

void NRCInterface::printStats()
{
    std::stringstream ss;
    Falcor::logInfo(ss.str());
}

void NRCInterface::reset()
{
    printf("[clear]--------------------------------------------------------------------------\n");
    nrc_Net_ref->reset();
}

void NRCInterface::registerNRCResources(
    Falcor::ref<Falcor::Buffer> pScreenQueryBuffer,
    Falcor::ref<Falcor::Buffer> pTrainingQueryBuffer,
    Falcor::ref<Falcor::Buffer> pTrainingSampleBuffer,
    Falcor::ref<Falcor::Buffer> pSharedCounterBuffer,
    Falcor::ref<Falcor::Buffer> pInferenceRadiancePixel,
    Falcor::ref<Falcor::Texture> pScreenResultTexture
)
{
    if (this == nullptr)
    {
        Falcor::logWarning("this is Empty!\n");
        exit(-1);
    }
    if (pScreenResultTexture.get() == nullptr)
    {
        Falcor::logWarning("pScreenResultTexture is Empty!\n");
        exit(-1);
    }
    cudaResources.screenResult = Falcor::cuda_utils::mapTextureToSurface(pScreenResultTexture, cudaArrayColorAttachment);
    cudaResources.screenQuery = (NRC::inputBase*)pScreenQueryBuffer.get()->getCudaMemory()->getMappedData();
    cudaResources.trainingQuery = (NRC::inputBase*)pTrainingQueryBuffer.get()->getCudaMemory()->getMappedData();
    cudaResources.trainingSample = (NRC::trainSample*)pTrainingSampleBuffer.get()->getCudaMemory()->getMappedData();
    cudaResources.inferenceQueryPixel = (uint2*)pInferenceRadiancePixel.get()->getCudaMemory()->getMappedData();
    uint32_t* counterBuffer = (uint32_t*)pSharedCounterBuffer.get()->getCudaMemory()->getMappedData();
    cudaResources.counterBufferPtr = counterBuffer;
    cudaResources.trainingSampleCounter = &counterBuffer[0];
    cudaResources.inferenceCounter = &counterBuffer[1];
    cudaResources.trainingQueryCounter = &counterBuffer[2];
}

} // namespace NRC
