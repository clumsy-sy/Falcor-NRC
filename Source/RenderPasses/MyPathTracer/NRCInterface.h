#pragma once

#include "NRCNetwork.h"

#include <Falcor.h>
#include <memory>
#include "Utils/CudaUtils.h"
#include "Utils/Logger.h"

namespace NRC
{

class NRCInterface
{
public:
    NRCInterface(Falcor::ref<Falcor::Device> pDevice);
    ~NRCInterface();

    void trainFrame(uint32_t train_cnt, uint32_t self_queriy_cnt, bool shuffle);

    void trainSampleFrame(uint32_t train_cnt, bool shuffle);

    void inferenceFrame(uint32_t infer_cnt, bool useRF);

    void printStats();

    void reset();

    void registerNRCResources(
        Falcor::ref<Falcor::Buffer> pScreenQueryBuffer,
        Falcor::ref<Falcor::Buffer> pTrainingQueryBuffer,
        Falcor::ref<Falcor::Buffer> pTrainingSampleBuffer,
        Falcor::ref<Falcor::Buffer> pSharedCounterBuffer,
        Falcor::ref<Falcor::Buffer> pInferenceRadiancePixel,
        Falcor::ref<Falcor::Texture> pScreenResultTexture
    );

    auto getDevice() -> Falcor::ref<Falcor::Device> { return pDevice; }

    std::shared_ptr<NRCNetwork> nrc_Net_ref;
    Falcor::ref<Falcor::Device> pDevice;

    struct
    {
        int n_frames = 0;
        float training_loss_avg = 0; // EMA
        const float ema_factor = 0.8f;
        const int print_every = 100;
    } mStats;

    // register interop texture/surface here
    struct
    {
        // cuda device pointers in unified memory space.
        NRC::inputBase* screenQuery = nullptr;
        cudaSurfaceObject_t screenResult; // write inferenced results here
        NRC::inputBase* trainingQuery = nullptr;
        NRC::trainSample* trainingSample = nullptr;
        ::uint2* inferenceQueryPixel = nullptr;
        uint32_t* counterBufferPtr = nullptr;
        uint32_t* trainingQueryCounter = nullptr;
        uint32_t* trainingSampleCounter = nullptr;
        uint32_t* inferenceCounter = nullptr;
    } cudaResources;
};
} // namespace NRC
