// =====================================================================================
// v1_kernels.cuh - V1's kernels, declared so the later versions can reuse the ones they
//                  do not replace. The definitions are in v1_basic.cu.
// Authors: Quách Châu Hạo Kiệt, Phan Thanh Bình, Tôn Thất Minh Đăng
// =====================================================================================
//
// Launching a kernel defined in another .cu file only needs its declaration (no
// relocatable device code), so e.g. V2 replaces the blur and Sobel kernels with tiled ones
// and keeps calling V1's harrisResponseKernel, nmsCompactKernel and warpBlendKernel.
#pragma once

#include "gpu_common.cuh"

// Stage 1 (owner: Quách Châu Hạo Kiệt)
__global__ void rgbToGrayKernel(const uint8_t* rgb, float* gray, int w, int h);
__global__ void blurRowsKernel(const float* in, float* out, int w, int h, const float* k, int radius);
__global__ void blurColsKernel(const float* in, float* out, int w, int h, const float* k, int radius);

// Stage 2 (owners: Tôn Thất Minh Đăng for Sobel and the response, Phan Thanh Bình for NMS and top-K)
__global__ void sobelProductsKernel(const float* img, int w, int h, Params p, float* ixx, float* iyy, float* ixy);
__global__ void harrisResponseKernel(const float* sxx, const float* syy, const float* sxy, float* R, int w, int h,
                                     float k);
__global__ void nmsCompactKernel(const float* R, int w, int h, int border, int nmsRadius, Keypoint* cand,
                                 int capacity, int* count);
void selectTopK(Keypoint* cand, int n, cudaStream_t stream = 0);

// Stage 6 (owner: Tôn Thất Minh Đăng)
__global__ void warpBlendKernel(const uint8_t* A, int wA, int hA, const uint8_t* B, int wB, int hB, Mat3f Hi, int ox,
                                int oy, uint8_t* out, int outW, int outH);
