#pragma once

#include <cuda.h>
#include <curand.h>
#include "RadianceStructure.slang"

namespace NRC
{

__device__ float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ float3 operator*(float3 a, float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator/(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ float3 safe_div(float3 a, float3 b)
{
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

__device__ void safe_num(float3& num)
{
    num.x = isinf(num.x) || isnan(num.x) ? 0 : num.x;
    num.y = isinf(num.y) || isnan(num.y) ? 0 : num.y;
    num.z = isinf(num.z) || isnan(num.z) ? 0 : num.z;
}

template<typename T = float>
__global__ void check_nans(uint32_t n_elements, T* data)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_elements)
        return;
    if (isnan(data[i]) || isinf(data[i]))
    {
        data[i] = (T)0.f;
    }
}

void printInputBase(const inputBase base)
{
    printf("|hit_pos : {%3.2f %3.2f %3.2f}|  ", base.hit_pos.x, base.hit_pos.y, base.hit_pos.z);
    printf("|sca_dir : {%3.2f %3.2f}|  ", base.scatter_dir.x, base.scatter_dir.y);
    printf("|suf_nor : {%3.2f %3.2f}|  ", base.suf_normal.x, base.suf_normal.y);
}

void printSample(const trainSample sample)
{
    printf("[%u]|radiance : {%3.2f %3.2f %3.2f}|  ", sample.train_idx, sample.L.x, sample.L.y, sample.L.z);
    printf("|thp : {%3.2f %3.2f %3.2f}|  --  ", sample.thp.x, sample.thp.y, sample.thp.z);
    printf("|hit_pos : {%3.2f %3.2f %3.2f}|  ", sample.input.hit_pos.x, sample.input.hit_pos.y, sample.input.hit_pos.z);
    printf("|sca_dir : {%3.2f %3.2f}|  ", sample.input.scatter_dir.x, sample.input.scatter_dir.y);
    printf("|suf_nor : {%3.2f %3.2f}|  ", sample.input.suf_normal.x, sample.input.suf_normal.y);
}

template<typename T>
void showMsg(const T* dataOnDevice, int size)
{
    T* dataOnHost = new T[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        std::cout << dataOnHost[i] << " ";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

uint32_t showMsg_counter(uint32_t* dataOnDevice)
{
    uint32_t* dataOnHost = new uint32_t[1];
    cudaMemcpy(dataOnHost, dataOnDevice, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("%u\n", dataOnHost[0]);
    uint32_t res = dataOnHost[0];
    delete[] dataOnHost;
    return res;
}

void showMsgColor(const float3* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    float3* dataOnHost = new float3[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printf("{%4.2f %4.2f %4.2f}\n", dataOnHost[i].x, dataOnHost[i].y, dataOnHost[i].z);
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showMsgBase(const inputBase* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    inputBase* dataOnHost = new inputBase[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(inputBase), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printInputBase(dataOnHost[i]);
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showMsgSample(const trainSample* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    trainSample* dataOnHost = new trainSample[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(trainSample), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printSample(dataOnHost[i]);
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

} // namespace NRC
