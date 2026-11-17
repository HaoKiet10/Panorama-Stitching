// =====================================================================================
// v3_kernels.cuh - V3's texture warp and its host helpers, declared so the later versions
//                  can reuse them. The definitions are in v3_texture.cu.
// Author: Tôn Thất Minh Đăng
// =====================================================================================
#pragma once

#include "gpu_common.cuh"

// Stage 6 with B in a texture and H^-1 in V3's constant memory (owner: Tôn Thất Minh Đăng).
__global__ void warpBlendTexKernel(const uint8_t* A, int wA, int hA, cudaTextureObject_t texB, int wB, int hB,
                                   int ox, int oy, uint8_t* out, int outW, int outH);

// Host helpers (provided).
// Copies H^-1 (9 floats, row-major) into V3's constant memory.
void uploadInverseH(const float Hi[9], cudaStream_t stream = 0);
// Texture of the RGB device image d_rgb (w x h): hardware bilinear, clamp to edge, colours
// as 0..1 floats. Destroy it with cudaDestroyTextureObject and free arr with cudaFreeArray.
cudaTextureObject_t makeTexture(const uint8_t* d_rgb, int w, int h, cudaArray_t& arr, cudaStream_t stream = 0);
