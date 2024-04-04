# Falcor NRC

基于 [Falcor](https://github.com/NVIDIAGameWorks/Falcor) 和 [Neural Radiance Caching](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) 的动态全局光照方案

## 编译 & 运行

1. `cmd` 执行 `setup_vs2022.bat`
2. `vs` 打开 `build\windows-vs2022\Falcor.sln`
3. 右键 `Mogwai` 设为启动项目，编译运行即可

更多细节详见 [Falcor 教程](https://github.com/NVIDIAGameWorks/Falcor/blob/master/docs/index.md)

> 注意
> 1. falcor 不再自带 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)，所以本项目是主动添加的
> 2. `tcnn` 库中的 `fmt` 与原项目的 `fmt` 有冲突，已在 `setup_vs2022.bat` 中解决，更多细节详见 `tools/fix-tcnn-XXX.txt`
> 3. 本项目根目录 `CMakeList.txt` 中的 `CMAKE_CUDA_ARCHITECTURES` 需根据自己的显卡架构进行更改
