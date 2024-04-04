from falcor import *

def render_graph_MyPathTracer():
    g = RenderGraph("MyPathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    MyPathTracer = createPass("MyPathTracer")
    g.addPass(MyPathTracer, "MyPathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MyPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "MyPathTracer.viewW")
    g.addEdge("MyPathTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

MyPathTracer = render_graph_MyPathTracer()
try: m.addGraph(MyPathTracer)
except NameError: None
