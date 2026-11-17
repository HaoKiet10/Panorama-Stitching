// =====================================================================================
// gpu_common.cuh - small pieces shared by all GPU versions (V1, V2, ...).
// Authors: Quách Châu Hạo Kiệt, Phan Thanh Bình, Tôn Thất Minh Đăng
// Drafted with an AI assistant (Claude) and reviewed by the team; see the AI declaration
// in the notebook.
// =====================================================================================
//
// Include from .cu files only. Everything here is either a type or static, so every file
// that includes it gets its own copy and nothing clashes at link time.
#pragma once

#include "versions.h"
#include "cuda_utils.cuh"
#include "v0_stages.h"

#include <algorithm>
#include <chrono>

// 2-D launch configuration for the per-pixel kernels: 32 x 8 = 256 threads, rows of 32
// threads so neighbouring threads read neighbouring pixels (coalesced).
static const dim3 BLOCK(32, 8);
static inline dim3 gridFor(int w, int h) { return dim3((w + BLOCK.x - 1) / BLOCK.x, (h + BLOCK.y - 1) / BLOCK.y); }

// A 3x3 float matrix passed to a kernel by value (it lands in the kernel's parameter space,
// so no copy or device buffer is needed).
struct Mat3f {
    float m[9];
};

// Device comparator with V0's keypoint order (customKpComp in common.h): score high to low,
// then y, then x. Used to sort the keypoint candidates on the GPU with Thrust.
struct KeypointOrder {
    __host__ __device__ bool operator()(const Keypoint& a, const Keypoint& b) const { return customKpComp(a, b); }
};

// CPU timing for the stages that stay on the CPU (3, 4, 5), like V0.
using Clock = std::chrono::high_resolution_clock;
static inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Upper bound on the number of NMS survivors: with V0's tie rule two survivors of the 3x3
// NMS are never neighbours, so at most one pixel in every 2x2 block survives.
static inline int nmsCapacity(int w, int h) { return ((w + 1) / 2) * ((h + 1) / 2); }
