// =====================================================================================
// v3_texture.cu - V3 = V2 + stage 6 samples image B through texture memory (hardware
//                 bilinear filtering), with the inverse homography in constant memory.
// Author: Tôn Thất Minh Đăng (warpBlendTexKernel)
// Host code drafted with an AI assistant (Claude) and reviewed by the team; see the AI
// declaration in the notebook.
// =====================================================================================
//
// Optimization 2 of the GPU plan. In V1/V2 every output pixel reads 4 neighbours of B from
// global memory at an arbitrary position (wherever H^-1 lands), then does the bilinear
// weighting in software. Here B is bound to a texture: the texture cache is built for this
// kind of 2-D locality, out-of-range coordinates are clamped by the hardware, and one
// tex2D call returns the already-interpolated colour. Hi is the same for every thread, so it
// sits in constant memory (one broadcast read per warp).
//
// Changed from V2:  warpBlendTexKernel (stage 6), plus the host code that builds the texture
// Reused:           everything else (V2's tiled stencils, V1's other kernels)
//
// Precision: the texture unit computes the bilinear weights with 8 bits of fraction, so
// some panorama pixels differ from V0 by 1-2 grey levels. H and the keypoints are not
// affected (stage 6 is last). Compare the panoramas with a tolerance, not exactly.
//
// What is done and what is left:
//   * DONE  the host code, including the RGBA copy of B, the CUDA array and the texture.
//   * TODO  the body of warpBlendTexKernel. Until it is written the panorama is black.
#include "gpu_common.cuh"
#include "v1_kernels.cuh"
#include "v2_kernels.cuh"
#include "v3_kernels.cuh"   // declarations of this file's kernel and helpers, reused by V4+

// H^-1 (A's frame -> B's frame) as floats, like V0's warpAndBlend. Constant memory belongs
// to this file, so other versions fill it with uploadInverseH.
__constant__ float c_Hi[9];

// =====================================================================================
// Stage 6 - warp + blend with a texture
// Owner: Tôn Thất Minh Đăng
// =====================================================================================

// TODO: V1's warpBlendKernel with two changes, one thread per output pixel (c, r):
//   1. Hi comes from constant memory:  applyHf(c_Hi, px, py, qx, qy)
//   2. B's colour comes from the texture instead of 4 global reads:
//        float4 v = tex2D<float4>(texB, qx + 0.5f, qy + 0.5f);   // texel centres are at +0.5
//        colourB = (v.x, v.y, v.z) * 255.0f                      // RGBA, read as 0..1 floats
//      Keep V0's test for "inside B" (0 <= qx < wB and 0 <= qy < hB) and V0's edgeDistance
//      weight; the clamping of the 4 neighbours is done by the texture (address mode clamp).
//   A is read from global memory exactly as in V1, and the three cases (both, only A, only
//   B) and the rounding are the same as V1. out starts black.
__global__ void warpBlendTexKernel(const uint8_t* A, int wA, int hA, cudaTextureObject_t texB, int wB, int hB,
                                   int ox, int oy, uint8_t* out, int outW, int outH) {
}

// =====================================================================================
// Host code (provided)
// =====================================================================================

// RGB -> RGBA (alpha 255): textures need 1, 2 or 4 channels, not 3. Provided.
__global__ void rgbToRgbaKernel(const uint8_t* rgb, uchar4* rgba, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t i = (size_t)y * w + x;
    rgba[i] = make_uchar4(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255);
}

// Copies H^-1 into c_Hi.
void uploadInverseH(const float Hi[9], cudaStream_t stream) {
    CHECK(cudaMemcpyToSymbolAsync(c_Hi, Hi, 9 * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
}

// Builds a texture of the RGB image d_rgb (w x h) that returns bilinearly interpolated
// colours as 0..1 floats, with clamp-to-edge outside the image. The caller destroys the
// texture and frees the array.
cudaTextureObject_t makeTexture(const uint8_t* d_rgb, int w, int h, cudaArray_t& arr, cudaStream_t stream) {
    uchar4* d_rgba;
    CHECK(cudaMalloc(&d_rgba, (size_t)w * h * sizeof(uchar4)));
    rgbToRgbaKernel<<<gridFor(w, h), BLOCK, 0, stream>>>(d_rgb, d_rgba, w, h);
    CHECK(cudaGetLastError());

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    CHECK(cudaMallocArray(&arr, &fmt, w, h));
    CHECK(cudaMemcpy2DToArrayAsync(arr, 0, 0, d_rgba, w * sizeof(uchar4), w * sizeof(uchar4), h,
                                   cudaMemcpyDeviceToDevice, stream));
    CHECK(cudaStreamSynchronize(stream));  // the copy must finish before d_rgba is freed
    CHECK(cudaFree(d_rgba));

    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc tex = {};
    tex.addressMode[0] = cudaAddressModeClamp;      // like clampInt on the 4 neighbours
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;          // hardware bilinear interpolation
    tex.readMode = cudaReadModeNormalizedFloat;     // uchar 0..255 -> float 0..1
    tex.normalizedCoords = 0;                       // coordinates in pixels
    cudaTextureObject_t obj = 0;
    CHECK(cudaCreateTextureObject(&obj, &res, &tex, nullptr));
    return obj;
}

void runV3(const ImageU8& A, const ImageU8& B, const Params& p, Result& result, Timing& t) {
    t = Timing();
    const Clock::time_point t_all = Clock::now();
    GpuTimer timer;

    const std::vector<float> blurK = gaussianKernel(p.blur_sigma, p.blur_radius);
    const std::vector<float> harrisK = gaussianKernel(p.harris_radius / 3.0f, p.harris_radius);
    const std::vector<OrbPair> pattern = makeOrbPattern(p.orb_seed, p.orb_radius);

    const ImageU8* images[2] = {&A, &B};
    uint8_t* d_rgb[2] = {nullptr, nullptr};
    ImageF blurred[2];
    std::vector<Keypoint> kps[2];

    // ---- copies to the device: V2's weights into constant memory, both images
    for (int i = 0; i < 2; i++) CHECK(cudaMalloc(&d_rgb[i], images[i]->data.size()));
    timer.Start();
    uploadBlurWeights(blurK, harrisK);
    for (int i = 0; i < 2; i++)
        CHECK(cudaMemcpy(d_rgb[i], images[i]->data.data(), images[i]->data.size(), cudaMemcpyHostToDevice));
    timer.Stop();
    t.h2d_ms += timer.Elapsed();

    // ---- stages 1-2 on the GPU: unchanged from V2
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

        timer.Start();
        rgbToGrayKernel<<<gridFor(w, h), BLOCK>>>(d_rgb[i], d_gray, w, h);
        CHECK(cudaGetLastError());
        blurTiledDevice(d_gray, d_tmp, d_blur, w, h, 0, p.blur_radius);
        timer.Stop();
        t.stage_ms[ST_PREPROCESS] += timer.Elapsed();

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

    // ---- stage 6 on the GPU with a texture for B and Hi in constant memory
    int ox, oy, outW, outH;
    computeCanvas(A.width, A.height, B.width, B.height, result.H, ox, oy, outW, outH);
    double Hi[9];
    const bool invertible = invertH(result.H, Hi);
    float Hif[9];
    for (int k = 0; k < 9; k++) Hif[k] = (float)Hi[k];

    timer.Start();
    uploadInverseH(Hif);
    timer.Stop();
    t.h2d_ms += timer.Elapsed();

    const size_t outBytes = (size_t)outW * outH * 3;
    uint8_t* d_out;
    CHECK(cudaMalloc(&d_out, outBytes));
    cudaArray_t arrB = nullptr;
    timer.Start();
    const cudaTextureObject_t texB = makeTexture(d_rgb[1], B.width, B.height, arrB);  // part of stage 6
    CHECK(cudaMemset(d_out, 0, outBytes));
    if (invertible) {
        warpBlendTexKernel<<<gridFor(outW, outH), BLOCK>>>(d_rgb[0], A.width, A.height, texB, B.width, B.height, ox,
                                                            oy, d_out, outW, outH);
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

    CHECK(cudaDestroyTextureObject(texB));
    CHECK(cudaFreeArray(arrB));
    CHECK(cudaFree(d_out));
    for (int i = 0; i < 2; i++) CHECK(cudaFree(d_rgb[i]));

    t.total_ms = msSince(t_all);
    result.num_kp_a = (int)kps[0].size();
    result.num_kp_b = (int)kps[1].size();
    result.num_matches = (int)matches.size();
    if (result.num_inliers == 0) fprintf(stderr, "warning: no homography found, panorama uses the identity\n");
}
