
Texture2D<float4> bias;
Texture2D<float4> factor;
Texture2D<float4> radiance;
RWTexture2D<float4> output;

cbuffer CB
{
    uint view;
}

enum viewMode
{
    DEFAULT = 0,
    NOT_REFERENCE = 1,
    ONLY_REFERENCE = 2,
    ONLY_FACTOR = 3,
    NRC_vs_NO = 4
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    switch (view)
    {
    case viewMode::DEFAULT:
        // output[pixel] = factor[pixel] * radiance[pixel] + bias[pixel];
        output[pixel] = factor[pixel] * radiance[pixel] + output[pixel];
        break;
    case viewMode::NOT_REFERENCE:
        output[pixel] = bias[pixel];
        // output[pixel] = output[pixel];
        break;
    case viewMode::ONLY_REFERENCE:
        output[pixel] = radiance[pixel];
        break;
    case viewMode::ONLY_FACTOR:
        output[pixel] = factor[pixel];
        break;
    case viewMode::NRC_vs_NO:
        output[pixel] = factor[pixel] * radiance[pixel];
        break;
    default:
        output[pixel] = float4(1.0, 0.0, 1.0, 1.0);
    }
}
