// =====================================================================================
// common.h - shared types, parameters and deterministic helpers used by EVERY version.
// Authors: <Member A>, <Member B>, <Member C>
// Drafted with an AI assistant (Claude) and reviewed by the team; see the AI declaration
// in the notebook.
// =====================================================================================
//
// Why this file exists
// --------------------
// The spec requires every version (V0, V1, V2, ...) to use the same methods and the same
// parameters, and to produce results consistent with V0. Anything that must be IDENTICAL
// across versions therefore lives here, in one place:
//
//   * the data types passed between stages      (ImageU8, ImageF, Keypoint, Descriptor, Match)
//   * the parameters of every stage             (Params)
//   * the timing and result records             (Timing, Result)
//   * the ORB sampling pattern                  (makeOrbPattern)
//   * the RANSAC random samples                 (makeRansacSamples)
//   * small homography utilities                (applyH, invertH, cornerError)
//
// The random parts (ORB pattern, RANSAC samples) are generated ONCE on the host by a
// fixed-seed generator, then handed to the CPU or GPU code. This way V0 and V1 test exactly
// the same point pairs and exactly the same RANSAC hypotheses, so their outputs can be
// compared directly.
//
// Everything here can be included from .cpp and .cu files. The small helpers marked
// HOST_DEVICE can be called from host code AND from CUDA kernels.
#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

// __host__ __device__ when compiled by nvcc (.cu files), nothing for the plain C++ compiler
// (.cpp files such as v0_sequential.cpp), which does not know these keywords.
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// =====================================================================================
// Stages
// =====================================================================================
// Index of each pipeline stage. Used to store and print per-stage times (Timing::stage_ms).
enum Stage {
    ST_PREPROCESS = 0,  // 1. grayscale + Gaussian blur          (per image)
    ST_DETECT,          // 2. Harris + non-max suppression + top-K (per image)
    ST_DESCRIBE,        // 3. ORB orientation + rotated BRIEF      (per image)
    ST_MATCH,           // 4. Hamming matching + ratio test + cross-check
    ST_HOMOGRAPHY,      // 5. RANSAC + 4-point DLT, refit on inliers
    ST_WARP,            // 6. inverse warping + bilinear sampling + feathering
    NUM_STAGES
};

// Names printed in the timing table. The notebook parses the CSV line, not these names.
static const char* const STAGE_NAMES[NUM_STAGES] = {
    "1 preprocess", "2 detect", "3 describe", "4 match", "5 homography", "6 warp+blend"};

// =====================================================================================
// Data passed between stages
// =====================================================================================

// An 8-bit colour image as loaded from disk.
// Pixels are stored row by row, 3 bytes per pixel (R, G, B):
//   the red value of pixel (x, y) is data[(y * width + x) * 3 + 0]
struct ImageU8 {
    int width = 0, height = 0, channels = 0;  // channels is always 3 here
    std::vector<uint8_t> data;                // size = width * height * channels
};

// A single-channel float image (grayscale, blurred grayscale, Harris response...).
// The value of pixel (x, y) is data[y * width + x].
struct ImageF {
    int width = 0, height = 0;
    std::vector<float> data;                  // size = width * height
};

// A detected corner. Produced by stage 2, completed (angle) by stage 3.
struct Keypoint {
    int x, y;        // integer pixel position in the image
    float score;     // Harris response at (x, y); used to rank keypoints for top-K
    float angle;     // ORB orientation in radians, filled in by stage 3
};

// A 256-bit binary descriptor, stored as 8 words of 32 bits.
// Bit i (0..255) is (w[i / 32] >> (i % 32)) & 1.
static const int DESC_BITS = 256;
static const int DESC_WORDS = DESC_BITS / 32;
struct Descriptor { uint32_t w[DESC_WORDS]; };

// One accepted correspondence between the two images.
struct Match {
    int a;           // index into image A's keypoint/descriptor arrays
    int b;           // index into image B's keypoint/descriptor arrays
    int dist;        // Hamming distance between the two descriptors (0..256)
};

// =====================================================================================
// Parameters
// =====================================================================================
// One set of parameters shared by all versions. Tune them on the small images, then
// freeze them: the timing comparison is only fair if every version uses the same values.
// main.cu lets you override a few of them from the command line for experiments.
struct Params {
    // ---- 1. preprocessing
    int blur_radius = 2;         // kernel covers [-radius, +radius] -> 5 taps
    float blur_sigma = blur_radius / 3.0f;     // Gaussian standard deviation, in pixels

    // ---- 2. detection
    float harris_k = 0.04f;      // R = det(M) - k * trace(M)^2
    int harris_radius = 2;       // window for summing gradient products -> 5x5
    int nms_radius = 1;          // non-max suppression neighbourhood -> 3x3
    int max_keypoints = 2000;    // top-K: keep the K strongest corners
    int border = 16;             // ignore corners closer than this to the edge,
                                 // so the 31x31 ORB patch always fits in the image
    float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};    
    float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // ---- 3. description
    int orb_radius = 15;         // patch radius -> 31x31 patch
    uint32_t orb_seed = 12345;   // seed of the BRIEF sampling pattern

    // ---- 4. matching
    float ratio = 0.8f;          // Lowe ratio test: accept if best < ratio * second best
    bool cross_check = true;     // also require that the two keypoints pick each other

    // ---- 5. homography
    int ransac_iters = 1000;     // number of 4-point hypotheses
    float ransac_thresh = 3.0f;  // a match is an inlier if reprojection error < this (px)
    uint32_t ransac_seed = 2026; // seed of the RANSAC samples
};

// =====================================================================================
// Results
// =====================================================================================

// Times of ONE run of the pipeline, in milliseconds. main.cu averages several runs.
// Image loading and saving are done by main and are never included.
struct Timing {
    double stage_ms[NUM_STAGES] = {0, 0, 0, 0, 0, 0};  // per stage (A and B summed for 1-3)
    double h2d_ms = 0;          // host -> device copies (GPU versions only)
    double d2h_ms = 0;          // device -> host copies (GPU versions only)
    double total_ms = 0;        // end to end, from the start to the end of runVN
};

// What a run produces. main.cu prints the counts and H, and saves the panorama.
struct Result {
    ImageU8 panorama;                           // the stitched output image
    double H[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // homography, row-major 3x3, maps a point
                                                // of image B into image A's frame
    int num_kp_a = 0, num_kp_b = 0;             // keypoints kept after top-K
    int num_matches = 0;                        // matches after ratio test and cross-check
    int num_inliers = 0;                        // matches consistent with the final H
};

// =====================================================================================
// Deterministic random numbers
// =====================================================================================
// A tiny linear congruential generator (the "Numerical Recipes" constants).
// We use it instead of std::mt19937 + std::uniform_int_distribution because the standard
// distributions are allowed to give different numbers on different compilers; this one
// gives the same sequence everywhere, on the CPU and (if ever needed) on the GPU.
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed ? seed : 1u) {}

    // Next raw 32-bit value.
    uint32_t next() { s = s * 1664525u + 1013904223u; return s; }

    // Uniform integer in [0, n). Uses the high bits, which are the good bits of an LCG.
    uint32_t below(uint32_t n) { return (uint32_t)(((uint64_t)next() * n) >> 32); }
};

// =====================================================================================
// ORB sampling pattern (used by stage 3)
// =====================================================================================
// One BRIEF test compares the brightness at two offsets around a keypoint:
//   bit = I(p + (x1, y1)) < I(p + (x2, y2))
// The 256 offset pairs are fixed for the whole program.
struct OrbPair { int8_t x1, y1, x2, y2; };

// Generates the 256 offset pairs. Every point lies inside a circle of radius
// (orb_radius - 2), so after rotating it by any angle and rounding, it still fits in the
// patch (and, with Params::border, inside the image).
inline std::vector<OrbPair> makeOrbPattern(uint32_t seed, int orb_radius) {
    Lcg rng(seed);
    const int r = orb_radius - 2;
    std::vector<OrbPair> pattern;
    pattern.reserve(DESC_BITS);
    while ((int)pattern.size() < DESC_BITS) {
        int p[4];  // x1, y1, x2, y2
        for (int k = 0; k < 4; k += 2) {
            do {  // rejection sampling: draw in the square until the point is in the circle
                p[k] = (int)rng.below(2 * r + 1) - r;
                p[k + 1] = (int)rng.below(2 * r + 1) - r;
            } while (p[k] * p[k] + p[k + 1] * p[k + 1] > r * r);
        }
        if (p[0] == p[2] && p[1] == p[3]) continue;  // comparing a pixel with itself is useless
        pattern.push_back({(int8_t)p[0], (int8_t)p[1], (int8_t)p[2], (int8_t)p[3]});
    }
    return pattern;
}

// =====================================================================================
// RANSAC samples (used by stage 5)
// =====================================================================================
// Returns iters * 4 match indices: hypothesis `it` uses matches
//   samples[it*4 + 0], samples[it*4 + 1], samples[it*4 + 2], samples[it*4 + 3]
// The 4 indices of one hypothesis are always different.
// Returns an empty vector if there are fewer than 4 matches (no homography possible).
// Generating them on the host lets the GPU test exactly the same hypotheses as V0.
inline std::vector<int> makeRansacSamples(int num_matches, int iters, uint32_t seed) {
    std::vector<int> samples;
    if (num_matches < 4) return samples;
    samples.resize((size_t)iters * 4);
    Lcg rng(seed);
    for (int it = 0; it < iters; it++) {
        int* s = &samples[(size_t)it * 4];
        for (int j = 0; j < 4; j++) {
            bool again;
            do {  // redraw until different from the indices already chosen
                s[j] = (int)rng.below((uint32_t)num_matches);
                again = false;
                for (int k = 0; k < j; k++) again |= (s[k] == s[j]);
            } while (again);
        }
    }
    return samples;
}

// =====================================================================================
// Small numeric helpers (usable on both host and device)
// =====================================================================================

// Helper: clamp v into [lo, hi]. Use it to read pixels outside the image as the nearest edge
// pixel ("clamp to edge"), e.g. img.data[clampInt(y, 0, h - 1) * w + clampInt(x, 0, w - 1)].
HOST_DEVICE inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Helper: round to the nearest integer; an exact .5 always rounds up (floor(v + 0.5)).
// Use THIS everywhere a coordinate is rounded (e.g. rotated ORB offsets), on the CPU and the
// GPU: lroundf, rintf and __float2int_rn round .5 differently and would make V0 and V1 differ.
HOST_DEVICE inline int roundToInt(float v) { return (int)floorf(v + 0.5f); }

// Helper: clamp v into [lo, hi] for floats. Usable on both host and device.
HOST_DEVICE inline float clampFloat(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// Helper: find distance from a pixel to edge, used in stage 6 to get weights for both images in blending.
HOST_DEVICE inline float edgeDistance(float x, float y, float w, float h) {
    float dx = fminf(x + 1.0f, w - x);
    float dy = fminf(y + 1.0f, h - y);
    return fminf(dx, dy);
}

// =====================================================================================
// Homography helpers (host side)
// =====================================================================================
// A homography is a 3x3 matrix H stored row-major in 9 doubles:
//   | H[0] H[1] H[2] |
//   | H[3] H[4] H[5] |
//   | H[6] H[7] H[8] |
// It maps a point (x, y) to (x', y') = ((H0 x + H1 y + H2) / w, (H3 x + H4 y + H5) / w)
// with w = H6 x + H7 y + H8.

// Applies H to the point (x, y); the result is written to (ox, oy).
inline void applyH(const double H[9], double x, double y, double& ox, double& oy) {
    double w = H[6] * x + H[7] * y + H[8];
    ox = (H[0] * x + H[1] * y + H[2]) / w;
    oy = (H[3] * x + H[4] * y + H[5]) / w;
}

// Inverts a 3x3 matrix with the adjugate formula (inverse = adjugate / determinant).
// Returns false if the matrix is (nearly) singular. Stage 6 needs H^-1 for inverse warping.
inline bool invertH(const double H[9], double Hi[9]) {
    double det = H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) +
                 H[2] * (H[3] * H[7] - H[4] * H[6]);
    if (fabs(det) < 1e-12) return false;
    double inv = 1.0 / det;
    Hi[0] = (H[4] * H[8] - H[5] * H[7]) * inv;
    Hi[1] = (H[2] * H[7] - H[1] * H[8]) * inv;
    Hi[2] = (H[1] * H[5] - H[2] * H[4]) * inv;
    Hi[3] = (H[5] * H[6] - H[3] * H[8]) * inv;
    Hi[4] = (H[0] * H[8] - H[2] * H[6]) * inv;
    Hi[5] = (H[2] * H[3] - H[0] * H[5]) * inv;
    Hi[6] = (H[3] * H[7] - H[4] * H[6]) * inv;
    Hi[7] = (H[1] * H[6] - H[0] * H[7]) * inv;
    Hi[8] = (H[0] * H[4] - H[1] * H[3]) * inv;
    return true;
}

// Measures how far apart two homographies are, in pixels:
// maps image B's four corners with H and with a reference G and averages the distances.
// Used for the synthetic-pair check (G = ground truth, --gt) and to check that a GPU
// version agrees with V0 (G = V0's H, --ref-h). Below ~2 px is a good result.
inline double cornerError(const double H[9], const double G[9], int width, int height) {
    const double cx[4] = {0, (double)width, 0, (double)width};
    const double cy[4] = {0, 0, (double)height, (double)height};
    double sum = 0;
    for (int i = 0; i < 4; i++) {
        double hx, hy, gx, gy;
        applyH(H, cx[i], cy[i], hx, hy);
        applyH(G, cx[i], cy[i], gx, gy);
        sum += sqrt((hx - gx) * (hx - gx) + (hy - gy) * (hy - gy));
    }
    return sum / 4;
}

// Custom comparison function for sorting keypoints
inline bool customKpComp(const Keypoint& a, const Keypoint& b) {
    if(a.score != b.score) {
        return a.score > b.score;
    }
    if(a.y != b.y) {
        return a.y < b.y;
    }
    return a.x < b.x;
}