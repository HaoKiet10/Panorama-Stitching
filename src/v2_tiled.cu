// =====================================================================================
// v2_tiled.cu - V2 = V1 + shared-memory tiling for the stencils of stages 1 and 2, with
//               the filter weights in constant memory.
// Authors: Quách Châu Hạo Kiệt (tiled blur), Tôn Thất Minh Đăng (tiled Sobel)
// Host code drafted with an AI assistant (Claude) and reviewed by the team; see the AI
// declaration in the notebook.
// =====================================================================================
//
// Optimization 1 of the GPU plan. In V1 every thread of the blur reads its 5 neighbours
// from global memory, so each pixel is read 5 times per pass (9 times for Sobel). Here a
// block first loads its tile of the image plus a halo of `radius` pixels into shared
// memory, once, and then every thread reads its neighbours from shared memory. The 5 (or
// 2 * 5) weights are the same for every thread, so they live in constant memory, which
// broadcasts one value to the whole warp in a single read.
//
// Changed from V1:  blurRowsTiledKernel, blurColsTiledKernel, sobelProductsTiledKernel
//                   (used for stage 1 AND for the three Harris-window blurs of stage 2)
// Reused from V1:   rgbToGrayKernel, harrisResponseKernel, nmsCompactKernel, selectTopK,
//                   warpBlendKernel (v1_kernels.cuh)
//
// What is done and what is left:
//   * DONE  the host code (same as V1 except the launches below and the constant memory).
//   * TODO  the bodies of the 3 tiled kernels. Until they are written V2 finds no keypoints.
// The results must be identical to V1's: tiling changes where the data is read from, not
// the arithmetic (same weights, same clamping, same summation order).
#include "gpu_common.cuh"
#include "v1_kernels.cuh"

// The two Gaussian kernels in constant memory: c_K[0] = stage 1 blur, c_K[1] = Harris
// window. Each has 2 * radius + 1 weights; MAX_TAPS allows radius <= 7.
#define MAX_TAPS 15
__constant__ float c_K[2][MAX_TAPS];

// =====================================================================================
// Tiled separable blur (stage 1, and the Harris window in stage 2)
// Owner: Quách Châu Hạo Kiệt
// =====================================================================================

// TODO: horizontal pass with a shared-memory tile. Block = BLOCK (32 x 8) threads.
//   extern __shared__ float tile[];   // (blockDim.x + 2 * radius) * blockDim.y floats
//   1. Load: the block covers columns x0 .. x0 + blockDim.x - 1 of its rows; the tile row
//      also needs `radius` pixels on each side. Each thread loads its own pixel and the
//      threads near the left/right edge of the block also load the halo pixels, with
//      clampInt on the image column (exactly V1's clamp-to-edge). Rows outside the image
//      load nothing (those threads must still reach __syncthreads).
//   2. __syncthreads();
//   3. Threads inside the image compute
//        out(x, y) = sum over i = -radius..radius of c_K[kIdx][i + radius] * tile(tx + radius + i, ty)
//      summed in the same order as V1/V0.
__global__ void blurRowsTiledKernel(const float* in, float* out, int w, int h, int kIdx, int radius) {
}

// TODO: vertical pass, the same with a (blockDim.x) x (blockDim.y + 2 * radius) tile and
//   the halo above and below the block (clampInt on the row).
__global__ void blurColsTiledKernel(const float* in, float* out, int w, int h, int kIdx, int radius) {
}

// =====================================================================================
// Tiled Sobel + gradient products (stage 2)
// Owner: Tôn Thất Minh Đăng
// =====================================================================================

// TODO: V1's sobelProductsKernel with a (blockDim.x + 2) x (blockDim.y + 2) shared tile
//   (a 1-pixel halo on every side, clamp-to-edge when loading, including the 4 corners).
//   Then the same 3x3 sums as V1 (p.sobelX / p.sobelY, rows then columns) read from the
//   tile, and ixx = gx * gx, iyy = gy * gy, ixy = gx * gy.
__global__ void sobelProductsTiledKernel(const float* img, int w, int h, Params p, float* ixx, float* iyy, float* ixy) {
}

// =====================================================================================
// Host code (provided)
// =====================================================================================

// Separable tiled blur: in -> tmp (rows) -> out (columns), weights c_K[kIdx].
// The third launch parameter is the dynamic shared memory size of each block.
static void blurTiledDevice(const float* in, float* tmp, float* out, int w, int h, int kIdx, int radius) {
    const size_t rowTile = (BLOCK.x + 2 * radius) * BLOCK.y * sizeof(float);
    const size_t colTile = BLOCK.x * (BLOCK.y + 2 * radius) * sizeof(float);
    blurRowsTiledKernel<<<gridFor(w, h), BLOCK, rowTile>>>(in, tmp, w, h, kIdx, radius);
    CHECK(cudaGetLastError());
    blurColsTiledKernel<<<gridFor(w, h), BLOCK, colTile>>>(tmp, out, w, h, kIdx, radius);
    CHECK(cudaGetLastError());
}

void runV2(const ImageU8& A, const ImageU8& B, const Params& p, Result& result, Timing& t) {
    t = Timing();
    const Clock::time_point t_all = Clock::now();
    GpuTimer timer;

    const std::vector<float> blurK = gaussianKernel(p.blur_sigma, p.blur_radius);
    const std::vector<float> harrisK = gaussianKernel(p.harris_radius / 3.0f, p.harris_radius);
    const std::vector<OrbPair> pattern = makeOrbPattern(p.orb_seed, p.orb_radius);
    if ((int)blurK.size() > MAX_TAPS || (int)harrisK.size() > MAX_TAPS) {
        fprintf(stderr, "V2: blur or Harris radius too large for MAX_TAPS (%d)\n", MAX_TAPS);
        exit(EXIT_FAILURE);
    }

    const ImageU8* images[2] = {&A, &B};
    uint8_t* d_rgb[2] = {nullptr, nullptr};
    ImageF blurred[2];
    std::vector<Keypoint> kps[2];

    // ---- copies to the device: the weights into constant memory, both images
    for (int i = 0; i < 2; i++) CHECK(cudaMalloc(&d_rgb[i], images[i]->data.size()));
    timer.Start();
    CHECK(cudaMemcpyToSymbol(c_K, blurK.data(), blurK.size() * sizeof(float), 0));
    CHECK(cudaMemcpyToSymbol(c_K, harrisK.data(), harrisK.size() * sizeof(float), MAX_TAPS * sizeof(float)));
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

        // stage 1: grayscale (V1), then the tiled blur
        timer.Start();
        rgbToGrayKernel<<<gridFor(w, h), BLOCK>>>(d_rgb[i], d_gray, w, h);
        CHECK(cudaGetLastError());
        blurTiledDevice(d_gray, d_tmp, d_blur, w, h, 0, p.blur_radius);
        timer.Stop();
        t.stage_ms[ST_PREPROCESS] += timer.Elapsed();

        // stage 2: tiled Sobel, tiled Harris window, then V1's response, NMS and sort
        timer.Start();
        const size_t sobelTile = (BLOCK.x + 2) * (BLOCK.y + 2) * sizeof(float);
        sobelProductsTiledKernel<<<gridFor(w, h), BLOCK, sobelTile>>>(d_blur, w, h, p, d_ixx, d_iyy, d_ixy);
        CHECK(cudaGetLastError());
        blurTiledDevice(d_ixx, d_tmp, d_sxx, w, h, 1, p.harris_radius);
        blurTiledDevice(d_iyy, d_tmp, d_syy, w, h, 1, p.harris_radius);
        blurTiledDevice(d_ixy, d_tmp, d_sxy, w, h, 1, p.harris_radius);
        harrisResponseKernel<<<gridFor(w, h), BLOCK>>>(d_sxx, d_syy, d_sxy, d_R, w, h, p.harris_k);
        CHECK(cudaGetLastError());
        CHECK(cudaMemset(d_count, 0, sizeof(int)));
        nmsCompactKernel<<<gridFor(w, h), BLOCK>>>(d_R, w, h, p.border, p.nms_radius, d_cand, capacity, d_count);
        CHECK(cudaGetLastError());
        int count = 0;
        CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        count = std::min(count, capacity);
        selectTopK(d_cand, count);
        timer.Stop();
        t.stage_ms[ST_DETECT] += timer.Elapsed();

        // copies back: the blurred image and the top-K keypoints
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

    // ---- stage 6 on the GPU: unchanged from V1
    int ox, oy, outW, outH;
    computeCanvas(A.width, A.height, B.width, B.height, result.H, ox, oy, outW, outH);
    double Hi[9];
    const bool invertible = invertH(result.H, Hi);
    Mat3f Hif;
    for (int k = 0; k < 9; k++) Hif.m[k] = (float)Hi[k];

    const size_t outBytes = (size_t)outW * outH * 3;
    uint8_t* d_out;
    CHECK(cudaMalloc(&d_out, outBytes));
    timer.Start();
    CHECK(cudaMemset(d_out, 0, outBytes));
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

    t.total_ms = msSince(t_all);
    result.num_kp_a = (int)kps[0].size();
    result.num_kp_b = (int)kps[1].size();
    result.num_matches = (int)matches.size();
    if (result.num_inliers == 0) fprintf(stderr, "warning: no homography found, panorama uses the identity\n");
}
