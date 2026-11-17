// =====================================================================================
// v2_kernels.cuh - V2's tiled kernels and helpers, declared so the later versions can
//                  reuse them. The definitions are in v2_tiled.cu.
// Authors: Quách Châu Hạo Kiệt, Tôn Thất Minh Đăng
// =====================================================================================
#pragma once

#include "gpu_common.cuh"

// Constant memory holds up to MAX_TAPS weights per Gaussian kernel (radius <= 7).
#define MAX_TAPS 15

// Tiled stencils (owners: Quách Châu Hạo Kiệt for the blur, Tôn Thất Minh Đăng for Sobel).
// kIdx picks the weights: 0 = stage 1 blur, 1 = Harris window.
__global__ void blurRowsTiledKernel(const float* in, float* out, int w, int h, int kIdx, int radius);
__global__ void blurColsTiledKernel(const float* in, float* out, int w, int h, int kIdx, int radius);
__global__ void sobelProductsTiledKernel(const float* img, int w, int h, Params p, float* ixx, float* iyy, float* ixy);

// Host helpers (provided).
// Copies the two Gaussian kernels into V2's constant memory. Call once per run before
// any blurTiledDevice.
void uploadBlurWeights(const std::vector<float>& blurK, const std::vector<float>& harrisK);
// Separable tiled blur in -> tmp (rows) -> out (columns) with the weights c_K[kIdx].
void blurTiledDevice(const float* in, float* tmp, float* out, int w, int h, int kIdx, int radius,
                     cudaStream_t stream = 0);
