// =====================================================================================
// v1_basic.cu - V1: stages 1, 2 and 6 on the GPU with straightforward kernels (global
//               memory only); stages 3, 4 and 5 run V0's code on the CPU.
// Authors: Quách Châu Hạo Kiệt (host code, stage 1), Tôn Thất Minh Đăng (stage 2 Sobel and
//          Harris response, stage 6), Phan Thanh Bình (stage 2 NMS and top-K)
// Host code drafted with an AI assistant (Claude) and reviewed by the team; see the AI
// declaration in the notebook.
// =====================================================================================
//
// Data flow of one run (see the GPU plan in the notebook):
//
//   A, B --H2D--> [GPU] 1 gray + blur --> 2 Harris + NMS + top-K
//                                   |
//                 D2H: blurred images + keypoints
//                                   v
//                 [CPU] 3 ORB --> 4 match --> 5 RANSAC        (V0's functions, v0_stages.h)
//                                   |
//                 H2D: nothing but H (A and B are still on the GPU)
//                                   v
//                 [GPU] 6 warp + blend --D2H--> panorama
//
// What is done and what is left:
//   * DONE  the host code: device buffers, copies, the order of the kernels, timing, the
//           CPU stages, the canvas. Read runV1 at the bottom first.
//   * TODO  the bodies of the 7 kernels and selectTopK (each has its owner and the V0
//           function it must reproduce). Until they are written V1 runs, finds no
//           keypoints and saves a black panorama of A's size.
//
// Matching V0: every kernel should compute exactly what the V0 function computes, in the
// same order (same weights, same clamping, same loop order for sums, same tie rules), so
// that V1 finds the same keypoints and the same H. Check with:
//   ./stitch L R out0.png --version 0 --save-h h0.txt
//   ./stitch L R out1.png --version 1 --ref-h h0.txt        (corner error should be < 1 px)
// nvcc fuses a*b + c into one FMA instruction, which rounds differently from the CPU; if
// the keypoints differ only slightly, rebuild with FLAGS="-O3 -std=c++17 -fmad=false" to
// confirm that FMA is the cause.
//
// Timing: GPU work is timed with GpuTimer (CUDA events), the CPU stages with
// std::chrono, copies go to h2d_ms / d2h_ms. cudaMalloc / cudaFree are inside the
// end-to-end time but outside every stage.
#include "gpu_common.cuh"   // BLOCK, gridFor, Mat3f, KeypointOrder, msSince, nmsCapacity
#include "v1_kernels.cuh"    // declarations of this file's kernels, reused by V2+

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// =====================================================================================
// Stage 1 - grayscale + separable Gaussian blur
// Owner: Quách Châu Hạo Kiệt
// =====================================================================================

// TODO: one thread per pixel (x, y); threads outside the image return.
//   Same formula and order as V0's toGray:  gray = R * .299f + G * .587f + B * .114f
//   rgb is w * h * 3 bytes (RGB interleaved), gray is w * h floats.
__global__ void rgbToGrayKernel(const uint8_t* rgb, float* gray, int w, int h) {
}

// TODO: horizontal pass of V0's gaussianBlur, one thread per pixel:
//   out(x, y) = sum over i = -radius..radius of k[i + radius] * in(clampInt(x + i, 0, w - 1), y)
//   Sum in the same order as V0 (i from -radius up) into a local float.
//   k is the device copy of gaussianKernel(...), 2 * radius + 1 floats.
//   Stage 2 reuses this kernel for the Harris window, so keep it generic.
__global__ void blurRowsKernel(const float* in, float* out, int w, int h, const float* k, int radius) {
}

// TODO: vertical pass, the same with clampInt(y + i, 0, h - 1).
__global__ void blurColsKernel(const float* in, float* out, int w, int h, const float* k, int radius) {
}

// =====================================================================================
// Stage 2 - Harris response, non-max suppression, top-K
// Owners: Tôn Thất Minh Đăng (Sobel, response), Phan Thanh Bình (NMS, top-K)
// =====================================================================================

// TODO (Đăng): step 1 of V0's harrisResponse, one thread per pixel:
//   gx, gy from the 3x3 Sobel filters p.sobelX / p.sobelY with clamp-to-edge, summed in V0's
//   order (i over rows, then j over columns); then ixx = gx * gx, iyy = gy * gy, ixy = gx * gy.
__global__ void sobelProductsKernel(const float* img, int w, int h, Params p, float* ixx, float* iyy, float* ixy) {
}

// TODO (Đăng): last step of V0's harrisResponse, one thread per pixel:
//   R = (sxx * syy - sxy * sxy) - k * (sxx + syy)^2       (same expression order as V0)
//   sxx, syy, sxy are the gradient products after the Gaussian window (the host runs
//   blurRowsKernel / blurColsKernel on each of them, like V0's three gaussianBlur calls).
__global__ void harrisResponseKernel(const float* sxx, const float* syy, const float* sxy, float* R, int w, int h,
                                     float k) {
}

// TODO (Bình): step 1 of V0's selectKeypoints, one thread per pixel:
//   - skip pixels closer than border to any edge, and R <= 0
//   - keep the pixel only if it is the maximum of its (2 * nmsRadius + 1)^2 neighbourhood,
//     with V0's tie rule (an equal neighbour wins if its index y * w + x is smaller)
//   - survivors: slot = atomicAdd(count, 1); if slot < capacity, cand[slot] = {x, y, R, 0.0f}
//   The atomic makes the order of cand random; selectTopK's sort makes it deterministic.
__global__ void nmsCompactKernel(const float* R, int w, int h, int border, int nmsRadius, Keypoint* cand,
                                 int capacity, int* count) {
}

// TODO (Bình): sort the n candidates on the device, strongest first, in V0's order, on the
//   given stream (V1 uses the default stream 0; V4 runs A and B on two streams):
//   thrust::sort(thrust::cuda::par.on(stream), cand, cand + n, KeypointOrder());
//   KeypointOrder (gpu_common.cuh) is V0's customKpComp. The host then copies the first
//   min(n, max_keypoints) back, which is V0's top-K cut.
void selectTopK(Keypoint* cand, int n, cudaStream_t stream) {
    (void)cand;
    (void)n;
    (void)stream;
}

// =====================================================================================
// Stage 6 - inverse warping + bilinear sampling + feathering
// Owner: Tôn Thất Minh Đăng
// =====================================================================================

// TODO (Đăng): the per-pixel loop of V0's warpAndBlend, one thread per output pixel (c, r):
//   (px, py) = (c + ox, r + oy) in A's frame; (qx, qy) = applyHf(Hi.m, px, py) in B's frame;
//   colour from A (if inside), bilinear colour from B (if inside), edgeDistance weights,
//   the same three cases as V0 (both, only A, only B). out is outW * outH * 3 bytes and
//   starts black, so pixels covered by neither image can simply return.
__global__ void warpBlendKernel(const uint8_t* A, int wA, int hA, const uint8_t* B, int wB, int hB, Mat3f Hi, int ox,
                                int oy, uint8_t* out, int outW, int outH) {
}

// =====================================================================================
// Host code (provided)
// =====================================================================================

// Separable blur on the device: in -> tmp (rows) -> out (columns). Used by stages 1 and 2.
static void blurDevice(const float* in, float* tmp, float* out, int w, int h, const float* d_k, int radius) {
    blurRowsKernel<<<gridFor(w, h), BLOCK>>>(in, tmp, w, h, d_k, radius);
    CHECK(cudaGetLastError());
    blurColsKernel<<<gridFor(w, h), BLOCK>>>(tmp, out, w, h, d_k, radius);
    CHECK(cudaGetLastError());
}

void runV1(const ImageU8& A, const ImageU8& B, const Params& p, Result& result, Timing& t) {
    t = Timing();
    const Clock::time_point t_all = Clock::now();
    GpuTimer timer;

    // Same host-side inputs as V0: the blur kernel (also the Harris window, which V0 builds
    // with sigma = harris_radius / 3) and the ORB pattern.
    const std::vector<float> blurK = gaussianKernel(p.blur_sigma, p.blur_radius);
    const std::vector<float> harrisK = gaussianKernel(p.harris_radius / 3.0f, p.harris_radius);
    const std::vector<OrbPair> pattern = makeOrbPattern(p.orb_seed, p.orb_radius);

    const ImageU8* images[2] = {&A, &B};
    uint8_t* d_rgb[2] = {nullptr, nullptr};  // A and B stay on the device until stage 6
    float *d_blurK = nullptr, *d_harrisK = nullptr;
    ImageF blurred[2];
    std::vector<Keypoint> kps[2];

    // ---- copies to the device: both images and the two filter kernels
    CHECK(cudaMalloc(&d_blurK, blurK.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_harrisK, harrisK.size() * sizeof(float)));
    for (int i = 0; i < 2; i++) CHECK(cudaMalloc(&d_rgb[i], images[i]->data.size()));
    timer.Start();
    CHECK(cudaMemcpy(d_blurK, blurK.data(), blurK.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_harrisK, harrisK.data(), harrisK.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < 2; i++)
        CHECK(cudaMemcpy(d_rgb[i], images[i]->data.data(), images[i]->data.size(), cudaMemcpyHostToDevice));
    timer.Stop();
    t.h2d_ms += timer.Elapsed();

    // ---- stages 1-2 on the GPU, one image at a time
    for (int i = 0; i < 2; i++) {
        const int w = images[i]->width, h = images[i]->height;
        const size_t n = (size_t)w * h;
        const int capacity = nmsCapacity(w, h);

        float *d_gray, *d_tmp, *d_blur, *d_ixx, *d_iyy, *d_ixy, *d_sxx, *d_syy, *d_sxy, *d_R;
        Keypoint* d_cand;
        int* d_count;
        float** bufs[] = {&d_gray, &d_tmp, &d_blur, &d_ixx, &d_iyy, &d_ixy, &d_sxx, &d_syy, &d_sxy, &d_R};
        for (float** b : bufs) CHECK(cudaMalloc(b, n * sizeof(float)));
        CHECK(cudaMalloc(&d_cand, (size_t)capacity * sizeof(Keypoint)));
        CHECK(cudaMalloc(&d_count, sizeof(int)));

        // stage 1: grayscale, then blur
        timer.Start();
        rgbToGrayKernel<<<gridFor(w, h), BLOCK>>>(d_rgb[i], d_gray, w, h);
        CHECK(cudaGetLastError());
        blurDevice(d_gray, d_tmp, d_blur, w, h, d_blurK, p.blur_radius);
        timer.Stop();
        t.stage_ms[ST_PREPROCESS] += timer.Elapsed();

        // stage 2: gradient products, Gaussian window, response, NMS + compaction, sort
        timer.Start();
        sobelProductsKernel<<<gridFor(w, h), BLOCK>>>(d_blur, w, h, p, d_ixx, d_iyy, d_ixy);
        CHECK(cudaGetLastError());
        blurDevice(d_ixx, d_tmp, d_sxx, w, h, d_harrisK, p.harris_radius);
        blurDevice(d_iyy, d_tmp, d_syy, w, h, d_harrisK, p.harris_radius);
        blurDevice(d_ixy, d_tmp, d_sxy, w, h, d_harrisK, p.harris_radius);
        harrisResponseKernel<<<gridFor(w, h), BLOCK>>>(d_sxx, d_syy, d_sxy, d_R, w, h, p.harris_k);
        CHECK(cudaGetLastError());
        CHECK(cudaMemset(d_count, 0, sizeof(int)));
        nmsCompactKernel<<<gridFor(w, h), BLOCK>>>(d_R, w, h, p.border, p.nms_radius, d_cand, capacity, d_count);
        CHECK(cudaGetLastError());
        int count = 0;
        CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));  // 4 bytes, part of stage 2
        count = std::min(count, capacity);
        selectTopK(d_cand, count);
        timer.Stop();
        t.stage_ms[ST_DETECT] += timer.Elapsed();

        // copies back: the blurred image (stage 3 reads it on the CPU) and the top-K keypoints
        const int k = std::min(count, p.max_keypoints);
        blurred[i].width = w;
        blurred[i].height = h;
        blurred[i].data.resize(n);
        kps[i].resize(k);
        timer.Start();
        CHECK(cudaMemcpy(blurred[i].data.data(), d_blur, n * sizeof(float), cudaMemcpyDeviceToHost));
        if (k > 0) CHECK(cudaMemcpy(kps[i].data(), d_cand, (size_t)k * sizeof(Keypoint), cudaMemcpyDeviceToHost));
        timer.Stop();
        t.d2h_ms += timer.Elapsed();

        for (float** b : bufs) CHECK(cudaFree(*b));
        CHECK(cudaFree(d_cand));
        CHECK(cudaFree(d_count));
    }

    // ---- stages 3-5 on the CPU: exactly V0's code
    std::vector<Descriptor> desc[2];
    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < 2; i++) describe(blurred[i], kps[i], pattern, p.orb_radius, desc[i]);
    t.stage_ms[ST_DESCRIBE] = msSince(t0);

    t0 = Clock::now();
    std::vector<Match> matches;
    matchDescriptors(desc[0], desc[1], p, matches);
    t.stage_ms[ST_MATCH] = msSince(t0);

    t0 = Clock::now();
    result.num_inliers = estimateHomography(kps[0], kps[1], matches, p, result.H);
    t.stage_ms[ST_HOMOGRAPHY] = msSince(t0);

    // ---- stage 6 on the GPU: canvas on the host (V0's computeCanvas), pixels on the device
    int ox, oy, outW, outH;
    computeCanvas(A.width, A.height, B.width, B.height, result.H, ox, oy, outW, outH);
    double Hi[9];
    const bool invertible = invertH(result.H, Hi);
    Mat3f Hif;
    for (int k = 0; k < 9; k++) Hif.m[k] = (float)Hi[k];  // float, like V0's warpAndBlend

    const size_t outBytes = (size_t)outW * outH * 3;
    uint8_t* d_out;
    CHECK(cudaMalloc(&d_out, outBytes));
    timer.Start();
    CHECK(cudaMemset(d_out, 0, outBytes));  // black where neither image covers the canvas
    if (invertible) {
        warpBlendKernel<<<gridFor(outW, outH), BLOCK>>>(d_rgb[0], A.width, A.height, d_rgb[1], B.width, B.height, Hif,
                                                         ox, oy, d_out, outW, outH);
        CHECK(cudaGetLastError());
    }
    timer.Stop();
    t.stage_ms[ST_WARP] = timer.Elapsed();

    result.panorama.width = outW;
    result.panorama.height = outH;
    result.panorama.channels = 3;
    result.panorama.data.resize(outBytes);
    timer.Start();
    CHECK(cudaMemcpy(result.panorama.data.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    timer.Stop();
    t.d2h_ms += timer.Elapsed();

    CHECK(cudaFree(d_out));
    for (int i = 0; i < 2; i++) CHECK(cudaFree(d_rgb[i]));
    CHECK(cudaFree(d_blurK));
    CHECK(cudaFree(d_harrisK));

    t.total_ms = msSince(t_all);
    result.num_kp_a = (int)kps[0].size();
    result.num_kp_b = (int)kps[1].size();
    result.num_matches = (int)matches.size();
    if (result.num_inliers == 0) fprintf(stderr, "warning: no homography found, panorama uses the identity\n");
}
