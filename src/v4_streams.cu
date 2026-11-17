// =====================================================================================
// v4_streams.cu - V4 = V3 + prefix-sum (scan) compaction of the NMS survivors instead of a
//                 global atomic counter, and two CUDA streams so images A and B run
//                 stages 1-2 at the same time.
// Author: Phan Thanh Bình (scan compaction kernels)
// Host code drafted with an AI assistant (Claude) and reviewed by the team; see the AI
// declaration in the notebook.
// =====================================================================================
//
// Optimization 3 of the GPU plan.
//   * Scan compaction. In V1-V3 every survivor does atomicAdd on ONE counter, so the
//     survivors are serialized on that address and land in a random order. Here every
//     pixel writes a flag (1 = survivor), an exclusive prefix sum of the flags gives each
//     survivor its slot, and a scatter writes the survivors in pixel order: no atomics,
//     and the candidate list is deterministic even before the sort.
//   * Streams. A and B are independent until stage 3, so each gets its own stream: while
//     one image's kernels leave SMs idle (small grids, the tail of a kernel), the other's
//     fill them, and one image's copies overlap the other's kernels. Asynchronous copies
//     need page-locked host memory, so the host buffers are registered with
//     cudaHostRegister (that cost is inside the end-to-end time, outside the stages).
//
// Changed from V3:  nmsFlagKernel, scanBlockKernel, addBlockSumsKernel, scatterKernel
//                   (replace V1's nmsCompactKernel), and the host code (two streams)
// Reused:           V1's grayscale, response and top-K sort, V2's tiled stencils,
//                   V3's texture warp
//
// Timing with streams: each stage is timed with CUDA events on its own stream and the two
// images are added, as in the other versions. Because A and B overlap, the stage times
// can add up to MORE than the end-to-end time; the end-to-end time shows the gain.
//
// What is done and what is left:
//   * DONE  the host code, including the recursive scan driver (scanDevice).
//   * TODO  the bodies of the 4 scan-compaction kernels. Until they are written V4 finds
//           no keypoints.
#include "gpu_common.cuh"
#include "v1_kernels.cuh"
#include "v2_kernels.cuh"
#include "v3_kernels.cuh"

// Threads per block of the scan kernels; each block scans SCAN_THREADS elements.
#define SCAN_THREADS 1024

// =====================================================================================
// Stage 2 - NMS + compaction by prefix sum
// Owner: Phan Thanh Bình
// =====================================================================================

// TODO: V1's nmsCompactKernel test, but every pixel (x, y) writes a flag instead of
//   appending: flags[y * w + x] = 1 if the pixel survives (inside the border, R > 0, maximum
//   of its neighbourhood with V0's tie rule), else 0. Every pixel must write, also the
//   border ones (0), because the scan reads all w * h flags.
__global__ void nmsFlagKernel(const float* R, int w, int h, int border, int nmsRadius, int* flags) {
}

// TODO: exclusive scan of one block's chunk. Block b covers in[b * SCAN_THREADS ..
//   b * SCAN_THREADS + SCAN_THREADS - 1], one element per thread (elements past n count as 0).
//   1. load into __shared__ int s[SCAN_THREADS]
//   2. scan in shared memory (Hillis-Steele with double buffering, or Blelloch's
//      up-sweep / down-sweep), with __syncthreads() between steps
//   3. out[i] = sum of the chunk's elements BEFORE i (exclusive), for i < n
//   4. thread 0 writes the chunk's total to blockSums[b]
__global__ void scanBlockKernel(const int* in, int* out, int* blockSums, int n) {
}

// TODO: out[i] += blockOffsets[i / SCAN_THREADS] for every i < n, where blockOffsets is the
//   exclusive scan of the block totals (so chunk b gets the sum of chunks 0 .. b - 1).
__global__ void addBlockSumsKernel(int* out, const int* blockOffsets, int n) {
}

// TODO: one thread per pixel i = y * w + x: if flags[i] is 1 and pos[i] < capacity, write
//   cand[pos[i]] = {x, y, R[i], 0.0f}. pos is the exclusive scan of flags, so the
//   survivors are packed in pixel order.
__global__ void scatterKernel(const int* flags, const int* pos, const float* R, int w, int h, Keypoint* cand,
                              int capacity) {
}

// =====================================================================================
// Host code (provided)
// =====================================================================================

// Exclusive prefix sum of n ints, d_in -> d_out, on `stream`. One level scans chunks of
// SCAN_THREADS; if there is more than one chunk, the chunk totals are scanned the same
// way (recursively) and added back. 3840 x 2160 pixels need 3 levels.
static void scanDevice(const int* d_in, int* d_out, int n, cudaStream_t stream) {
    const int blocks = (n + SCAN_THREADS - 1) / SCAN_THREADS;
    int* d_sums;
    CHECK(cudaMallocAsync(&d_sums, blocks * sizeof(int), stream));
    scanBlockKernel<<<blocks, SCAN_THREADS, 0, stream>>>(d_in, d_out, d_sums, n);
    CHECK(cudaGetLastError());
    if (blocks > 1) {
        int* d_offsets;
        CHECK(cudaMallocAsync(&d_offsets, blocks * sizeof(int), stream));
        scanDevice(d_sums, d_offsets, blocks, stream);
        addBlockSumsKernel<<<blocks, SCAN_THREADS, 0, stream>>>(d_out, d_offsets, n);
        CHECK(cudaGetLastError());
        CHECK(cudaFreeAsync(d_offsets, stream));
    }
    CHECK(cudaFreeAsync(d_sums, stream));
}

// Device buffers and timers of one image while stages 1-2 run on its stream.
struct ImageWork {
    int w = 0, h = 0, capacity = 0;
    size_t n = 0;
    uint8_t* d_rgb = nullptr;  // kept until stage 6
    float *d_gray, *d_tmp, *d_blur, *d_ixx, *d_iyy, *d_ixy, *d_sxx, *d_syy, *d_sxy, *d_R;
    int *d_flags, *d_pos;
    Keypoint* d_cand;
    int* h_last = nullptr;       // pinned: the last flag and the last scan value -> count
    Keypoint* h_kps = nullptr;   // pinned: the top-K keypoints
};

void runV4(const ImageU8& A, const ImageU8& B, const Params& p, Result& result, Timing& t) {
    t = Timing();
    const Clock::time_point t_all = Clock::now();

    const std::vector<float> blurK = gaussianKernel(p.blur_sigma, p.blur_radius);
    const std::vector<float> harrisK = gaussianKernel(p.harris_radius / 3.0f, p.harris_radius);
    const std::vector<OrbPair> pattern = makeOrbPattern(p.orb_seed, p.orb_radius);

    const ImageU8* images[2] = {&A, &B};
    cudaStream_t stream[2];
    ImageWork wk[2];
    ImageF blurred[2];
    std::vector<Keypoint> kps[2];

    // ---- setup (outside the stages): streams, buffers, page-locked host memory
    for (int i = 0; i < 2; i++) {
        CHECK(cudaStreamCreate(&stream[i]));
        ImageWork& W = wk[i];
        W.w = images[i]->width;
        W.h = images[i]->height;
        W.n = (size_t)W.w * W.h;
        W.capacity = nmsCapacity(W.w, W.h);
        CHECK(cudaMalloc(&W.d_rgb, images[i]->data.size()));
        float** bufs[] = {&W.d_gray, &W.d_tmp, &W.d_blur, &W.d_ixx, &W.d_iyy, &W.d_ixy, &W.d_sxx, &W.d_syy, &W.d_sxy, &W.d_R};
        for (float** b : bufs) CHECK(cudaMalloc(b, W.n * sizeof(float)));
        CHECK(cudaMalloc(&W.d_flags, W.n * sizeof(int)));
        CHECK(cudaMalloc(&W.d_pos, W.n * sizeof(int)));
        // zeroed once so the program stays safe (0 keypoints) while the scan kernels are TODO
        CHECK(cudaMemset(W.d_flags, 0, W.n * sizeof(int)));
        CHECK(cudaMemset(W.d_pos, 0, W.n * sizeof(int)));
        CHECK(cudaMalloc(&W.d_cand, (size_t)W.capacity * sizeof(Keypoint)));
        CHECK(cudaMallocHost(&W.h_last, 2 * sizeof(int)));
        CHECK(cudaMallocHost(&W.h_kps, (size_t)std::max(p.max_keypoints, 1) * sizeof(Keypoint)));
        blurred[i].width = W.w;
        blurred[i].height = W.h;
        blurred[i].data.resize(W.n);
        // const_cast: cudaHostRegister only pins the pages, it does not write to them
        CHECK(cudaHostRegister(const_cast<uint8_t*>(images[i]->data.data()), images[i]->data.size(),
                               cudaHostRegisterDefault));
        CHECK(cudaHostRegister(blurred[i].data.data(), W.n * sizeof(float), cudaHostRegisterDefault));
    }
    uploadBlurWeights(blurK, harrisK);  // V2's constant memory (synchronous, a few bytes)

    // ---- phase 1, both streams: copy in, stage 1, stage 2 up to the compaction
    GpuTimer tH2D[2] = {GpuTimer(stream[0]), GpuTimer(stream[1])};
    GpuTimer tS1[2] = {GpuTimer(stream[0]), GpuTimer(stream[1])};
    GpuTimer tS2a[2] = {GpuTimer(stream[0]), GpuTimer(stream[1])};
    for (int i = 0; i < 2; i++) {
        ImageWork& W = wk[i];
        cudaStream_t s = stream[i];
        const dim3 grid = gridFor(W.w, W.h);

        tH2D[i].Start();
        CHECK(cudaMemcpyAsync(W.d_rgb, images[i]->data.data(), images[i]->data.size(), cudaMemcpyHostToDevice, s));
        tH2D[i].Stop();

        tS1[i].Start();
        rgbToGrayKernel<<<grid, BLOCK, 0, s>>>(W.d_rgb, W.d_gray, W.w, W.h);
        CHECK(cudaGetLastError());
        blurTiledDevice(W.d_gray, W.d_tmp, W.d_blur, W.w, W.h, 0, p.blur_radius, s);
        tS1[i].Stop();

        tS2a[i].Start();
        const size_t sobelTile = (BLOCK.x + 2) * (BLOCK.y + 2) * sizeof(float);
        sobelProductsTiledKernel<<<grid, BLOCK, sobelTile, s>>>(W.d_blur, W.w, W.h, p, W.d_ixx, W.d_iyy, W.d_ixy);
        CHECK(cudaGetLastError());
        blurTiledDevice(W.d_ixx, W.d_tmp, W.d_sxx, W.w, W.h, 1, p.harris_radius, s);
        blurTiledDevice(W.d_iyy, W.d_tmp, W.d_syy, W.w, W.h, 1, p.harris_radius, s);
        blurTiledDevice(W.d_ixy, W.d_tmp, W.d_sxy, W.w, W.h, 1, p.harris_radius, s);
        harrisResponseKernel<<<grid, BLOCK, 0, s>>>(W.d_sxx, W.d_syy, W.d_sxy, W.d_R, W.w, W.h, p.harris_k);
        CHECK(cudaGetLastError());
        nmsFlagKernel<<<grid, BLOCK, 0, s>>>(W.d_R, W.w, W.h, p.border, p.nms_radius, W.d_flags);
        CHECK(cudaGetLastError());
        scanDevice(W.d_flags, W.d_pos, (int)W.n, s);
        scatterKernel<<<grid, BLOCK, 0, s>>>(W.d_flags, W.d_pos, W.d_R, W.w, W.h, W.d_cand, W.capacity);
        CHECK(cudaGetLastError());
        // count = last scan value + last flag; the 2 ints come back with the stream's other work
        CHECK(cudaMemcpyAsync(&W.h_last[0], W.d_flags + W.n - 1, sizeof(int), cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(&W.h_last[1], W.d_pos + W.n - 1, sizeof(int), cudaMemcpyDeviceToHost, s));
        tS2a[i].Stop();
    }

    // ---- phase 2, both streams: sort (needs the count on the host), copy back
    GpuTimer tS2b[2] = {GpuTimer(stream[0]), GpuTimer(stream[1])};
    GpuTimer tD2H[2] = {GpuTimer(stream[0]), GpuTimer(stream[1])};
    int k[2];
    for (int i = 0; i < 2; i++) {
        ImageWork& W = wk[i];
        cudaStream_t s = stream[i];
        CHECK(cudaStreamSynchronize(s));  // h_last is ready
        const int count = std::max(0, std::min(W.h_last[0] + W.h_last[1], W.capacity));
        k[i] = std::min(count, p.max_keypoints);

        tS2b[i].Start();
        selectTopK(W.d_cand, count, s);
        tS2b[i].Stop();

        tD2H[i].Start();
        CHECK(cudaMemcpyAsync(blurred[i].data.data(), W.d_blur, W.n * sizeof(float), cudaMemcpyDeviceToHost, s));
        if (k[i] > 0)
            CHECK(cudaMemcpyAsync(W.h_kps, W.d_cand, (size_t)k[i] * sizeof(Keypoint), cudaMemcpyDeviceToHost, s));
        tD2H[i].Stop();
    }
    for (int i = 0; i < 2; i++) {
        CHECK(cudaStreamSynchronize(stream[i]));
        kps[i].assign(wk[i].h_kps, wk[i].h_kps + k[i]);
        t.h2d_ms += tH2D[i].Elapsed();
        t.stage_ms[ST_PREPROCESS] += tS1[i].Elapsed();
        t.stage_ms[ST_DETECT] += tS2a[i].Elapsed() + tS2b[i].Elapsed();
        t.d2h_ms += tD2H[i].Elapsed();
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

    // ---- stage 6 on the GPU: V3's texture warp (one image, so one stream)
    int ox, oy, outW, outH;
    computeCanvas(A.width, A.height, B.width, B.height, result.H, ox, oy, outW, outH);
    double Hi[9];
    const bool invertible = invertH(result.H, Hi);
    float Hif[9];
    for (int q = 0; q < 9; q++) Hif[q] = (float)Hi[q];

    GpuTimer timer(stream[0]);
    timer.Start();
    uploadInverseH(Hif, stream[0]);
    timer.Stop();
    t.h2d_ms += timer.Elapsed();

    const size_t outBytes = (size_t)outW * outH * 3;
    uint8_t* d_out;
    CHECK(cudaMalloc(&d_out, outBytes));
    cudaArray_t arrB = nullptr;
    timer.Start();
    const cudaTextureObject_t texB = makeTexture(wk[1].d_rgb, B.width, B.height, arrB, stream[0]);
    CHECK(cudaMemsetAsync(d_out, 0, outBytes, stream[0]));
    if (invertible) {
        warpBlendTexKernel<<<gridFor(outW, outH), BLOCK, 0, stream[0]>>>(wk[0].d_rgb, A.width, A.height, texB,
                                                                          B.width, B.height, ox, oy, d_out, outW, outH);
        CHECK(cudaGetLastError());
    }
    timer.Stop();
    t.stage_ms[ST_WARP] = timer.Elapsed();

    result.panorama.width = outW;
    result.panorama.height = outH;
    result.panorama.channels = 3;
    result.panorama.data.resize(outBytes);
    timer.Start();
    CHECK(cudaMemcpyAsync(result.panorama.data.data(), d_out, outBytes, cudaMemcpyDeviceToHost, stream[0]));
    timer.Stop();
    t.d2h_ms += timer.Elapsed();  // Elapsed waits for the copy

    // ---- cleanup
    CHECK(cudaDestroyTextureObject(texB));
    CHECK(cudaFreeArray(arrB));
    CHECK(cudaFree(d_out));
    for (int i = 0; i < 2; i++) {
        ImageWork& W = wk[i];
        CHECK(cudaHostUnregister(const_cast<uint8_t*>(images[i]->data.data())));
        CHECK(cudaHostUnregister(blurred[i].data.data()));
        float* bufs[] = {W.d_gray, W.d_tmp, W.d_blur, W.d_ixx, W.d_iyy, W.d_ixy, W.d_sxx, W.d_syy, W.d_sxy, W.d_R};
        for (float* b : bufs) CHECK(cudaFree(b));
        CHECK(cudaFree(W.d_rgb));
        CHECK(cudaFree(W.d_flags));
        CHECK(cudaFree(W.d_pos));
        CHECK(cudaFree(W.d_cand));
        CHECK(cudaFreeHost(W.h_last));
        CHECK(cudaFreeHost(W.h_kps));
        CHECK(cudaStreamDestroy(stream[i]));
    }

    t.total_ms = msSince(t_all);
    result.num_kp_a = (int)kps[0].size();
    result.num_kp_b = (int)kps[1].size();
    result.num_matches = (int)matches.size();
    if (result.num_inliers == 0) fprintf(stderr, "warning: no homography found, panorama uses the identity\n");
}
