add_plugin(NRCPathTracer)

target_sources(NRCPathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    Params.slang
    PathState.slang
    PathTracer.slang
    NRCPathTracer.cpp
    NRCPathTracer.h
    PathTracerNRD.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
    NRCHelper.cuh
    NRCNetwork.h
    NRCNetwork.cu
    NRCInterface.h
    NRCInterface.cpp
    PathTracerNRC.slang
    Combination.cs.slang
)

set_target_properties(NRCPathTracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_copy_shaders(NRCPathTracer RenderPasses/NRCPathTracer)

target_source_group(NRCPathTracer "RenderPasses")
