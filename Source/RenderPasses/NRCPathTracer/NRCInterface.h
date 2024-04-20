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
    /**
     * @brief Call the train function
     */
    void train(uint32_t train_cnt, uint32_t self_queriy_cnt, bool shuffle);
    /**
     * @brief Call the simple train function
     */
    void trainSimple(uint32_t train_cnt, bool shuffle);
    /**
     * @brief Call the inference function
     */
    void inference(uint32_t infer_cnt, bool useRF);
    /**
     * @brief log some message
     */
    void log();
    /**
     * @brief reset network
     */
    void reset();
    /**
     * @brief map shader resources to cuda resources
     */
    void mapResources(
        Falcor::ref<Falcor::Buffer> pScreenQueryBuffer,
        Falcor::ref<Falcor::Buffer> pTrainQueryBuffer,
        Falcor::ref<Falcor::Buffer> pTrainSampleBuffer,
        Falcor::ref<Falcor::Buffer> pSharedCounterBuffer,
        Falcor::ref<Falcor::Buffer> pInferRadiancePixel,
        Falcor::ref<Falcor::Texture> pScreenResultTexture
    );
    /**
     * @brief get device message
     */
    auto getDevice() -> Falcor::ref<Falcor::Device> { return pDevice; }
    /**
     *
     */
    std::shared_ptr<NRCNetwork> getNetworkSPtr()
    {
        std::shared_ptr<NRCNetwork> tmp = nrc_network_ref;
        return tmp;
    }

private:
    std::shared_ptr<NRCNetwork> nrc_network_ref;
    Falcor::ref<Falcor::Device> pDevice;

    struct
    {
        int n_frames = 0;
        float train_loss_avg = 0; // EMA
        const float ema_factor = 0.8f;
    } mStats;

    // shader resource to cuda resource
    struct
    {
        // cuda device pointers in unified memory space.
        NRC::inputBase* screen_query = nullptr;
        NRC::inputBase* train_query = nullptr;
        NRC::trainSample* train_sample = nullptr;
        ::uint2* infer_query_pixel = nullptr;
        uint32_t* counter_buffer_ptr = nullptr;
        uint32_t* train_query_cnt = nullptr;
        uint32_t* train_sample_cnt = nullptr;
        uint32_t* infer_cnt = nullptr;
        cudaSurfaceObject_t screen_result; // results
    } mCudaResources;
};
} // namespace NRC
