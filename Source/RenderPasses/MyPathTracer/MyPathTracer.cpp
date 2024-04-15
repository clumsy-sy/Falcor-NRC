/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "MyPathTracer.h"
#include <cstdint>
#include "Core/API/FBO.h"
#include "Core/API/Formats.h"
#include "Core/Object.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Logger.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, MyPathTracer>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/MyPathTracer/MyPathTracer.rt.slang";
const char kCompositeShaderFile[] = "RenderPasses/MyPathTracer/Composite.cs.slang";

const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxInferenceBounces[] = "maxInferBounces";
const char kMaxTrainBounces[] = "maxTrainBounces";
const char kRefreshNRCNetwork[] = "refreshNRCNetwork";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
const char kUseNRCTraining[] = "useNRCTraining";

enum viewMode
{
    DEFAULT = 0,
    NOT_REFERENCE = 1,
    ONLY_REFERENCE = 2,
    ONLY_FACTOR = 3,
    DIFF = 4,
    NRC_vs_NO = 5,
    TEST = 6,
};

const Gui::DropdownList kNRCViewModeList = {
    {(uint)viewMode::DEFAULT, "with NRC"},
    {(uint)viewMode::NOT_REFERENCE, "NO NRC"},
    {(uint)viewMode::ONLY_REFERENCE, "ONLY NRC"},
    {(uint)viewMode::ONLY_FACTOR, "ONLY FACTOR"},
    {(uint)viewMode::DIFF, "DIFF"},
    {(uint)viewMode::NRC_vs_NO, "NRC VS NO NRC"},
    {(uint)viewMode::TEST, "just for test"},
};

} // namespace

MyPathTracer::MyPathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    // mNRC = FALCOR_ASSERT(mpSampleGenerator);
    mNRC.pNRC = std::make_shared<MININRC::NRCInterface>(pDevice);
    mNRC.pNetwork = mNRC.pNRC->getNetworkSPtr();
    if (mNRC.pNRC == nullptr || mNRC.pNetwork == nullptr)
    {
        logWarning("ERROR : NRC network init failed!\n");
        exit(-1);
    }
    mCompositePass = ComputePass::create(pDevice, kCompositeShaderFile, "main");
}

void MyPathTracer::initNRC(ref<Device> pDevice, Falcor::uint2 targetDim)
{
    if (!mNRC.pNRC.get() && !mNRC.pNetwork.get())
    {
        logWarning("pNRC or NRC NetWork not Exist!");
        exit(-1);
    }

    bool enableNRC = true;

    int max_training_bounces = 2; // max path segments for training suffix
    int mMaxTrainBounces = 0;
    // buffer
    mNRC.pTrainingRadianceQuery = mpDevice->createStructuredBuffer(
        sizeof(MININRC::inputBase),
        MININRC::max_train_query_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );
    mNRC.pTrainingtrainSample = mpDevice->createStructuredBuffer(
        sizeof(MININRC::trainSample),
        MININRC::max_train_sample_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );
    // we need record the last hit point
    mNRC.pTmpPathRecord = mpDevice->createStructuredBuffer(
        sizeof(MININRC::inputBase),
        MININRC::max_train_sample_size + MININRC::max_infer_query_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );
    mNRC.pInferenceRadianceQuery = mpDevice->createStructuredBuffer(
        sizeof(MININRC::inputBase),
        MININRC::max_infer_query_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );
    mNRC.pInferenceRadiancePixel = mpDevice->createStructuredBuffer(
        sizeof(Falcor::uint2),
        MININRC::max_infer_query_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );
    mNRC.pSharedCounterBuffer = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), 4, Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess
    );
    if (mNRC.pTrainingRadianceQuery == nullptr || mNRC.pTrainingtrainSample == nullptr || mNRC.pTmpPathRecord == nullptr ||
        mNRC.pInferenceRadianceQuery == nullptr || mNRC.pInferenceRadiancePixel == nullptr || mNRC.pSharedCounterBuffer == nullptr)
    {
        logWarning("pNRC buffer alloc failed");
        exit(-1);
    }
    // texture
    mNRC.pScreenResult = pDevice->createTexture2D(
        targetDim.x,
        targetDim.y,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        Falcor::ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mNRC.pScreenQueryFactor = pDevice->createTexture2D(
        targetDim.x,
        targetDim.y,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mNRC.pScreenQueryBias = pDevice->createTexture2D(
        targetDim.x,
        targetDim.y,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    if (mNRC.pScreenResult == nullptr || mNRC.pScreenQueryFactor == nullptr || mNRC.pScreenQueryBias == nullptr)
    {
        logWarning("pNRC texture alloc failed");
        exit(-1);
    }

    mNRC.pNRC->mapResources(
        mNRC.pInferenceRadianceQuery,
        mNRC.pTrainingRadianceQuery,
        mNRC.pTrainingtrainSample,
        mNRC.pSharedCounterBuffer,
        mNRC.pInferenceRadiancePixel,
        mNRC.pScreenResult
    );
}

void MyPathTracer::NRCEndFrame(ref<Device> pDevice)
{
    // buffer
    mNRC.pTrainingRadianceQuery->unmap();
    mNRC.pTrainingtrainSample->unmap();
    mNRC.pInferenceRadianceQuery->unmap();
    mNRC.pInferenceRadiancePixel->unmap();
    mNRC.pSharedCounterBuffer->unmap();
    if (mNRC.pTrainingRadianceQuery == nullptr || mNRC.pTrainingtrainSample == nullptr || mNRC.pInferenceRadianceQuery == nullptr ||
        mNRC.pInferenceRadiancePixel == nullptr || mNRC.pSharedCounterBuffer == nullptr)
    {
        logWarning("pNRC buffer alloc failed");
        exit(-1);
    }
    // texture
    mNRC.pScreenResult.reset();
}

/**
 * 解析属性信息，来自 py 文件
 */
void MyPathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxInferenceBounces)
            mNRC.mMaxInferBounces = value;
        else if (key == kMaxTrainBounces)
            mNRC.mMaxTrainBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in MyPathTracer properties.", key);
    }
}
Properties MyPathTracer::getProperties() const
{
    Properties props;
    props[kMaxInferenceBounces] = mNRC.mMaxInferBounces;
    props[kMaxTrainBounces] = mNRC.mMaxTrainBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
    // return {};
}

RenderPassReflection MyPathTracer::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    // reflector.addOutput("dst");
    // reflector.addInput("src");
    return reflector;
}

void MyPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    mTracer.pProgram->addDefine("NRC_MAX_TRAINING_BOUNCES", std::to_string(mNRC.mMaxTrainBounces));
    mTracer.pProgram->addDefine("NRC_MAX_INFERENCE_BOUNCES", std::to_string(mNRC.mMaxInferBounces));

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Get dimensions of ray dispatch.
    const Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // init nrc buffer
    if (mFrameCount == 0)
        initNRC(mpDevice, targetDim);
#ifdef LOG
    printf("[Frame] = %d \n", mFrameCount);
#endif
    pRenderContext->clearUAVCounter(mNRC.pTrainingRadianceQuery, 0);
    pRenderContext->clearUAVCounter(mNRC.pTrainingtrainSample, 0);
    pRenderContext->clearUAVCounter(mNRC.pTmpPathRecord, 0);
    pRenderContext->clearUAVCounter(mNRC.pInferenceRadianceQuery, 0);
    pRenderContext->clearUAVCounter(mNRC.pInferenceRadiancePixel, 0);
    pRenderContext->clearTexture(mNRC.pScreenResult.get());

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gWidth"] = targetDim.x;
    var["CB"]["gHeight"] = targetDim.y;
    var["CB"]["enableNRCTrain"] = mNRC.enableNRCTrain;
    var["CB"]["enableSelfQuery"] = mNRC.enableSelfQuery;
    var["CB"]["RADNDIVID"] = mNRC.train_ctrl.rand_ctrl;
    var["gTrainingRadianceQuery"] = mNRC.pTrainingRadianceQuery;
    var["gTrainingtrainSample"] = mNRC.pTrainingtrainSample;
    var["gTmpPathRecord"] = mNRC.pTmpPathRecord;
    var["gInferenceRadianceQuery"] = mNRC.pInferenceRadianceQuery;
    var["gInferenceRadiancePixel"] = mNRC.pInferenceRadiancePixel;
    var["gScreenQueryFactor"] = mNRC.pScreenQueryFactor;
    var["gScreenQueryBias"] = mNRC.pScreenQueryBias;

    if (mFrameCount == 0)
    {
        printf("[Scene Msg] : \n");
        printf(
            "------------: COMPUTE_DIRECT = %s  , USE_IMPORTANCE_SAMPLING = %s\n",
            mComputeDirect ? "true" : "false",
            mUseImportanceSampling ? "true" : "false"
        );
        printf(
            "------------: USE_ANALYTIC_LIGHTS = %s  , USE_EMISSIVE_LIGHTS = %s\n",
            mpScene->useAnalyticLights() ? "true" : "false",
            mpScene->useEmissiveLights() ? "true" : "false"
        );
        printf(
            "------------: USE_ENV_LIGHT = %s  , USE_ENV_BACKGROUND = %s\n",
            mpScene->useEnvLight() ? "true" : "false",
            mpScene->useEnvBackground() ? "true" : "false"
        );
    }

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);

    try
    {
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, Falcor::uint3(targetDim, 1));
    }
    catch (const std::exception& e)
    {
        std::cout << "[raytrace][error] : " << e.what() << std::endl;
        exit(-1);
    }
    if (mNRC.enableNRC)
    {
        //
        pRenderContext->copyBufferRegion(mNRC.pSharedCounterBuffer.get(), 0, mNRC.pTrainingtrainSample->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mNRC.pSharedCounterBuffer.get(), 4, mNRC.pInferenceRadianceQuery->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mNRC.pSharedCounterBuffer.get(), 8, mNRC.pTrainingRadianceQuery->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mNRC.pSharedCounterBuffer.get(), 12, mNRC.pTmpPathRecord->getUAVCounter().get(), 0, 4);
        auto element = mNRC.pSharedCounterBuffer->getElements<uint32_t>(0, 4);
        mNRC.pNetwork->copyCountToHost(element);
#ifdef LOG
        printf("laset rand control = %u ", mNRC.train_ctrl.rand_ctrl);
#endif
        // > 65535 too more train 4294967295 * 0.00001 =
        uint tmpTrainCnt = mNRC.pNetwork->nrc_counter.train_sample_cnt;
        // if(tmpTrainCnt > 65535 + 1000) {
        //     if(tmpTrainCnt - 65535 > 100000)
        //     {
        //         mNRC.train_ctrl.rand_ctrl -= 4294967;
        //     } else {
        //         mNRC.train_ctrl.rand_ctrl -= 429496;
        //     }
        // } else {
        //     mNRC.train_ctrl.rand_ctrl += 42949672;
        // }
        // if(mNRC.train_ctrl.rand_ctrl > 429496729)
        //     mNRC.train_ctrl.rand_ctrl = 4294967295 * 0.04;
        mNRC.train_ctrl.update(tmpTrainCnt);
        mNRC.train_ctrl.get_rand_ctrl();
#ifdef LOG
        printf("new rand control = %d tmpTrainCnt = %d\n", mNRC.train_ctrl.rand_ctrl, tmpTrainCnt);
        mNRC.train_ctrl.showParameters();
#endif

#ifdef LOG
        for (uint32_t i = 0; i < element.size(); i++)
        {
            printf("SharedBuffer[%i] = %u || ", i, element[i]);
        }
        printf("\n");
#endif
        //
        if (mNRC.enableNRCTrain)
        {
#ifdef LOG
            logInfo("[NRC] : train stage");
#endif
            if (mNRC.enableSelfQuery)
            {
                mNRC.pNRC->train(element[0], element[2], mNRC.enableShuffleTrain);
            }
            else
            {
                mNRC.pNRC->trainSimple(element[0], mNRC.enableShuffleTrain);
            }
            cudaDeviceSynchronize();
        }
        else
        {
#ifdef LOG
            logInfo("[NRC log] : no train");
#endif
        }
        //
#ifdef LOG
        logInfo("[NRC log] :inference stage");
#endif
        mNRC.pNRC->inference(element[1], mNRC.enableReflectanceFactorization);
        cudaDeviceSynchronize();
        //
        auto Compositevar = mCompositePass->getRootVar();
        Compositevar["CB"]["view"] = mNRC.visualizeMode;
        Compositevar["factor"] = mNRC.pScreenQueryFactor;
        Compositevar["bias"] = mNRC.pScreenQueryBias;
        Compositevar["radiance"] = mNRC.pScreenResult;
        Compositevar["output"] = renderData.getTexture("color");
#ifdef LOG
        logInfo("[computer Pass] running");
#endif
        mCompositePass->execute(pRenderContext, Falcor::uint3(targetDim, 1));
    }

    // if (mFrameCount == 1)
    // {
    //     exit(0);
    // }
    mFrameCount++;
}

void MyPathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max Infer bounces", mNRC.mMaxInferBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.var("Max Train bounces", mNRC.mMaxTrainBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.checkbox("Use NRC", mNRC.enableNRC);
    widget.tooltip("Use Neural Radiance Cache", true);

    if (mNRC.enableNRC)
    {
        dirty |= widget.dropdown("view", kNRCViewModeList, mNRC.visualizeMode);

        dirty |= widget.checkbox("Use Reflectance Factorization", mNRC.enableReflectanceFactorization);
        widget.tooltip("Use ReflectanceFactorization: result = radiance * (diffuse + specular)", true);

        dirty |= widget.checkbox("Use NRC network train", mNRC.enableNRCTrain);
        widget.tooltip("Use Neural Radiance Network train", true);

        if (mNRC.enableNRCTrain)
        {
            dirty |= widget.checkbox("Use NRC network shuffle", mNRC.enableShuffleTrain);
            widget.tooltip("Use Neural Radiance Network train with shuffle data, better use", true);

            dirty |= widget.checkbox("Use NRC network Self Query", mNRC.enableSelfQuery);
            widget.tooltip("Use Neural Radiance Network train with Self Query", true);
        }

        if (widget.button("reset NRC network"))
        {
            mNRC.pNRC->reset();
            dirty = true;
        }
        dirty |= widget.var("Learning rate", mNRC.pNetwork->getLearningRate(), 0.f, 1e-1f, 1e-5f);
    }

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

/**
 * 设置场景
 */
void MyPathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("MyPathTracer: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}
/**
 * 编译 shader
 */
void MyPathTracer::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}

void MyPathTracer::clearTexture(ref<Device> pDevice)
{
    if (mNRC.pScreenResult != nullptr)
    {
        mNRC.pScreenResult.reset();
    }
    mNRC.pScreenResult = pDevice->createTexture2D(
        1980,
        1080,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        Falcor::ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
}
