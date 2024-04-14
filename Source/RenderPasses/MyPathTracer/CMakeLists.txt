if(NOT FALCOR_HAS_CUDA)
    return()
endif()

add_plugin(MyPathTracer)

target_sources(MyPathTracer PRIVATE
    NRCInterface.h
    NRCInterface.cpp
    NRCHelper.cuh
    NRCNetwork.h
    NRCNetwork.cu
    RadianceStructure.slang
    NRCInterface.slang
    MyPathTracer.cpp
    MyPathTracer.h
    MyPathTracer.rt.slang
)
set_target_properties(MyPathTracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_copy_shaders(MyPathTracer RenderPasses/MyPathTracer)

target_source_group(MyPathTracer "RenderPasses")
