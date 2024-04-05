from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("NRCPathTracer")
    NRCPathTracer = createPass("NRCPathTracer", {'samplesPerPixel': 1})
    g.addPass(NRCPathTracer, "NRCPathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "NRCPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "NRCPathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "NRCPathTracer.mvec")
    g.addEdge("NRCPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

NRCPathTracer = render_graph_PathTracer()
try: m.addGraph(NRCPathTracer)
except NameError: None
