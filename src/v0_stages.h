// =====================================================================================
// v0_stages.h - the V0 stage functions that the GPU versions reuse on the CPU.
// Authors: Quách Châu Hạo Kiệt, Phan Thanh Bình, Tôn Thất Minh Đăng
// =====================================================================================
//
// V1 runs stages 1, 2 and 6 on the GPU and keeps stages 3, 4 and 5 on the CPU (see the GPU
// plan in the notebook). Instead of copying that code, the GPU versions call the V0
// functions declared here, so the CPU part of every version is exactly V0's.
//
// The definitions are in v0_sequential.cpp. Everything else in that file stays private
// (static) to V0.
#pragma once

#include "common.h"

// Stage 1: normalized 1-D Gaussian kernel, 2 * radius + 1 weights summing to 1.
std::vector<float> gaussianKernel(float sigma, int radius);

// Stage 3: ORB orientation + rotated BRIEF on the blurred grayscale image.
// Fills kps[k].angle and one 256-bit descriptor per keypoint.
void describe(const ImageF& img, std::vector<Keypoint>& kps, const std::vector<OrbPair>& pattern, int radius,
              std::vector<Descriptor>& desc);

// Stage 4: brute-force Hamming matching with the ratio test and cross-check.
void matchDescriptors(const std::vector<Descriptor>& dA, const std::vector<Descriptor>& dB, const Params& p,
                      std::vector<Match>& matches);

// Stage 5: RANSAC + normalized DLT. Writes H (B -> A), returns the inlier count (0 = none).
int estimateHomography(const std::vector<Keypoint>& kA, const std::vector<Keypoint>& kB,
                       const std::vector<Match>& matches, const Params& p, double H[9]);

// Stage 6, host part: the panorama canvas. Its top-left corner in A's frame is (ox, oy) and
// its size is width x height. GPU versions call this on the host so their canvas is
// exactly V0's; only the per-pixel loop moves to the GPU.
void computeCanvas(int wA, int hA, int wB, int hB, const double H[9], int& ox, int& oy, int& width, int& height);
