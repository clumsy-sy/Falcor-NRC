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
#pragma once

// use LOG
#define  LOG

#include <memory>
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "NRCInterface.h"

using namespace Falcor;

class MyPathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(MyPathTracer, "MyPathTracer", "Reference my path tracer.");

    static ref<MyPathTracer> create(ref<Device> pDevice, const Properties& props) { return make_ref<MyPathTracer>(pDevice, props); }

    MyPathTracer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    // virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void parseProperties(const Properties& props);
    void prepareVars();

    // 当前场景
    ref<Scene> mpScene;
    // 采样器
    ref<SampleGenerator> mpSampleGenerator;

    // Configuration
    // 是否计算直接光照
    bool mComputeDirect = true;
    // 是否对材质使用重要性采样
    bool mUseImportanceSampling = true;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;

    // Ray tracing program.
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;

    // NRC
    struct
    {
        std::shared_ptr<MININRC::NRCInterface> pNRC = nullptr;
        std::shared_ptr<MININRC::NRCNetwork> pNetwork = nullptr;

        bool enableNRC = true;
        bool enableNRCTrain = true;
        bool enableSelfQuery = false;
        bool enableShuffleTrain = true;
        bool enableReflectanceFactorization = true;

        uint mMaxTrainBounces = 4; // max path segments for training suffix
        uint mMaxInferBounces = 2;

        uint visualizeMode = 0;
        uint rand_ctrl = 4294967295 * 0.04;

        ref<Buffer> pTrainingRadianceQuery = nullptr;
        ref<Buffer> pTrainingtrainSample = nullptr;
        ref<Buffer> pInferenceRadianceQuery = nullptr;
        ref<Buffer> pInferenceRadiancePixel = nullptr;
        ref<Buffer> pTmpPathRecord = nullptr;
        ref<Buffer> pSharedCounterBuffer = nullptr;
        ref<Texture> pScreenResult = nullptr;

        ref<Texture> pScreenQueryFactor = nullptr;
        ref<Texture> pScreenQueryBias = nullptr;
        // ref<Texture> pScreenQueryReflectance = nullptr;

    } mNRC;

    ref<ComputePass> mCompositePass;

    void initNRC(ref<Device> pDevice, Falcor::uint2 targetDim);
    void NRCEndFrame(ref<Device> pDevice);
    void setNRCData(const RenderData& renderData);
    void clearTexture(ref<Device> pDevice);
};
