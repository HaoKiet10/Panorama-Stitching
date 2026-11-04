// =====================================================================================
// v0_sequential.cpp - V0: single-threaded reference implementation of the whole pipeline.
// Authors: <Member A> (stages 1, 5), <Member B> (stages 2, 6), <Member C> (stages 3, 4)
// =====================================================================================
//
// V0 is the reference for correctness and the baseline for every speedup, so:
//   * single thread only: no OpenMP, no std::thread, no hand-written SIMD (midterm rule);
//   * per-pixel maths in float, the precision the CUDA versions will use;
//   * deterministic: same input -> same output, every run, so V1+ can be compared with it.
//
// HOW THIS FILE IS ORGANISED
//   - helpers (provided):        timing, popcount, 3x3 multiply, 8x8 solver
//                                (clampInt and roundToInt are in common.h)
//   - stage functions (TODO):    one or more per stage, each with a comment describing the
//                                inputs, outputs and the algorithm step by step
//   - runV0 (provided):         calls the stages in order and times each one
//
// Every TODO function is empty (functions that return a value only have a bare `return`, so
// the file compiles). Until stage 6 is written the panorama is empty and saving it fails;
// that is expected. Fill the stages in order and check the printed counts as you go:
//   after stage 2: "keypoints A 2000, B 2000"        after stage 4: a few hundred matches
//   after stage 5: most matches are inliers, H looks like a shift of ~60% of the width
//
// If you adapt code from anywhere (papers, tutorials, open source), cite it right above the
// function AND in the notebook's References section.

#include "versions.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>

// =====================================================================================
// Helpers (provided)
// =====================================================================================

using Clock = std::chrono::high_resolution_clock;

// Milliseconds elapsed since t0. The spec asks for std::chrono::high_resolution_clock in V0.
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// clampInt and roundToInt live in common.h, shared with the CUDA versions.

// Number of 1-bits in a 32-bit word (portable; CUDA has __popc for the same job).
static inline int popcount32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
}

// C = A * B for row-major 3x3 matrices.
static void mat3Mul(const double A[9], const double B[9], double C[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
}

// Solves the 8x8 linear system M h = v (Gaussian elimination with partial pivoting).
// M and v are overwritten. Returns false if the system is (nearly) singular, which happens
// when the 4 RANSAC points are degenerate (e.g. 3 of them on one line).
static bool solve8(double M[8][8], double v[8], double h[8]) {
    for (int c = 0; c < 8; c++) {
        int piv = c;  // pick the row with the largest value in this column, for stability
        for (int r = c + 1; r < 8; r++)
            if (fabs(M[r][c]) > fabs(M[piv][c])) piv = r;
        if (fabs(M[piv][c]) < 1e-12) return false;
        if (piv != c) {
            for (int k = 0; k < 8; k++) std::swap(M[c][k], M[piv][k]);
            std::swap(v[c], v[piv]);
        }
        for (int r = c + 1; r < 8; r++) {  // eliminate below the pivot
            double f = M[r][c] / M[c][c];
            for (int k = c; k < 8; k++) M[r][k] -= f * M[c][k];
            v[r] -= f * v[c];
        }
    }
    for (int r = 7; r >= 0; r--) {  // back substitution
        double s = v[r];
        for (int k = r + 1; k < 8; k++) s -= M[r][k] * h[k];
        h[r] = s / M[r][r];
    }
    return true;
}

// =====================================================================================
// Stage 1 - preprocessing                                          owner: <Member A>
// =====================================================================================

// TODO: convert an RGB image to grayscale.
//   in   RGB image, 3 bytes per pixel
//   out  float image of the same size
//   For every pixel: gray = 0.299 R + 0.587 G + 0.114 B   (values stay in 0..255)
static void toGray(const ImageU8& in, ImageF& out) {
}

// TODO: build a normalized 1-D Gaussian kernel.
//   returns 2 * radius + 1 weights; weight i (i = -radius..radius) is proportional to
//   exp(-i^2 / (2 sigma^2)). Divide by the sum so the weights add up to 1
//   (otherwise the blur changes the image brightness).
static std::vector<float> gaussianKernel(float sigma, int radius) {
    return {};
}

// TODO: separable Gaussian blur.
//   in, out  float images of the same size;  k  the kernel from gaussianKernel
//   1. horizontal pass into a temporary image:  tmp(x, y) = sum_i k[i] * in(x + i, y)
//   2. vertical pass into out:                  out(x, y) = sum_i k[i] * tmp(x, y + i)
//   with i = -radius..radius and clamp-to-edge at the borders (clampInt).
//   Two 1-D passes give the same result as one 2-D k x k blur with 2k instead of k^2 work.
static void gaussianBlur(const ImageF& in, ImageF& out, const std::vector<float>& k, int radius) {
}

// =====================================================================================
// Stage 2 - keypoint detection                                     owner: <Member B>
// =====================================================================================

// TODO: Harris corner response for every pixel.
//   img  the blurred grayscale image;  R  output, one float per pixel (resize it to w * h)
//   1. Sobel gradients at every pixel (clamp-to-edge):
//        gx = [I(x+1,y-1) + 2 I(x+1,y) + I(x+1,y+1)] - [I(x-1,y-1) + 2 I(x-1,y) + I(x-1,y+1)]
//        gy = [I(x-1,y+1) + 2 I(x,y+1) + I(x+1,y+1)] - [I(x-1,y-1) + 2 I(x,y-1) + I(x+1,y-1)]
//      and store the products gx*gx, gy*gy, gx*gy in three arrays.
//   2. For every pixel, sum each product over the (2 r + 1)^2 window, r = p.harris_radius
//      (clamp-to-edge):  Sxx, Syy, Sxy.
//   3. R = (Sxx * Syy - Sxy^2) - p.harris_k * (Sxx + Syy)^2
//      R is large and positive at corners, negative on edges, near 0 in flat areas.
static void harrisResponse(const ImageF& img, const Params& p, std::vector<float>& R) {
}

// TODO: non-max suppression + top-K.
//   R  Harris response (w x h);  kps  output keypoints, strongest first
//   1. For every pixel at least p.border away from all four edges with R > 0:
//      keep it only if it is the maximum of its (2 * p.nms_radius + 1)^2 neighbourhood.
//      Ties: a neighbour with the SAME value beats the pixel only if it comes earlier in
//      row-major order (smaller y * w + x). This makes the result deterministic.
//   2. Push each survivor as {x, y, R, 0.0f}.
//   3. Sort by score, highest first; equal scores -> smaller y * w + x first.
//   4. Keep only the first p.max_keypoints.
//   The tie rules matter: the GPU version must pick exactly the same keypoints.
static void selectKeypoints(const std::vector<float>& R, int w, int h, const Params& p, std::vector<Keypoint>& kps) {
}

// =====================================================================================
// Stage 3 - description (ORB)                                      owner: <Member C>
// =====================================================================================

// TODO: ORB orientation of the keypoint at (cx, cy).
//   Over all offsets (dx, dy) with dx^2 + dy^2 <= radius^2 (a disc, not a square):
//     m10 += dx * I(cx + dx, cy + dy)
//     m01 += dy * I(cx + dx, cy + dy)
//   return atan2f(m01, m10)   -> the direction from the keypoint to the patch's
//                                 "centre of brightness", in radians.
//   No clamping needed: p.border keeps the disc inside the image. Accumulate in float.
static float orientation(const ImageF& img, int cx, int cy, int radius) {
    return 0.0f;
}

// TODO: rotated BRIEF descriptor for every keypoint.
//   img      the blurred grayscale image
//   kps      keypoints from stage 2; fill in kps[k].angle here
//   pattern  the 256 offset pairs from makeOrbPattern (common.h)
//   desc     output, one Descriptor per keypoint, all bits starting at 0
//   For each keypoint:
//     1. angle = orientation(img, x, y, radius);  c = cosf(angle), s = sinf(angle)
//     2. for each test i = 0..255 with offsets (x1, y1), (x2, y2):
//          rotate both:  (x', y') = (roundToInt(c x - s y), roundToInt(s x + c y))
//          v1 = I(kp + rotated offset 1),  v2 = I(kp + rotated offset 2)
//          if v1 < v2, set bit i:  desc[k].w[i >> 5] |= 1u << (i & 31)
//   Use roundToInt (common.h) so the GPU rounds exactly the same way.
static void describe(const ImageF& img, std::vector<Keypoint>& kps, const std::vector<OrbPair>& pattern, int radius, std::vector<Descriptor>& desc) {
}

// =====================================================================================
// Stage 4 - matching                                               owner: <Member C>
// =====================================================================================

// TODO: Hamming distance = number of differing bits = sum over the 8 words of
//   popcount32(a.w[k] ^ b.w[k]).
static inline int hamming(const Descriptor& a, const Descriptor& b) {
    return 0;
}

// TODO: brute-force matching with Lowe's ratio test and cross-check.
//   dA, dB   descriptors of image A and image B;  matches  output
//   1. Cross-check table (if p.cross_check): for every b in B, find the a in A with the
//      smallest distance (ties -> smallest a). Store it in best_of_b[b].
//   2. For every a in A, scan all of B and keep the best distance, the index of the best b
//      (ties -> smallest b) and the second-best distance.
//   3. Accept the pair (a, best_b) if
//        - ratio test:   best < p.ratio * second   (as floats; accept if B has only 1 point)
//        - cross-check:  best_of_b[best_b] == a    (skip if p.cross_check is false)
//      and push {a, best_b, best} in increasing order of a.
static void matchDescriptors(const std::vector<Descriptor>& dA, const std::vector<Descriptor>& dB, const Params& p, std::vector<Match>& matches) {
}

// =====================================================================================
// Stage 5 - homography (RANSAC + DLT)                              owner: <Member A>
// =====================================================================================

// TODO: Hartley normalization of n points (pts = x0, y0, x1, y1, ...).
//   Build T (3x3, row-major) that moves the points' centroid (cx, cy) to the origin and
//   scales them so their mean distance from it is sqrt(2):
//     s = sqrt(2) / mean distance       T = | s  0  -s cx |
//                                           | 0  s  -s cy |
//                                           | 0  0    1   |
//   Without it the DLT equations mix numbers ~1 and ~10^6 and lose precision.
static void normalizePoints(const double* pts, int n, double T[9]) {
}

// TODO: DLT - fit H (src -> dst) to n >= 4 point pairs, with H[8] fixed to 1.
//   src, dst  n points each, as x0, y0, x1, y1, ...;  H  output (row-major 3x3)
//   1. Ts = normalizePoints(src), Td = normalizePoints(dst); normalize every point.
//   2. Each pair (x, y) -> (u, v) gives two equations in the 8 unknowns h0..h7:
//        [x y 1 0 0 0 -u*x -u*y] . h = u
//        [0 0 0 x y 1 -v*x -v*y] . h = v
//      Accumulate the normal equations  M = sum(r r^T),  rhs = sum(r * u)  (and the same
//      for the v row). With n = 4 this gives the exact solution; with more pairs it is
//      the least-squares fit used for the refit on inliers.
//   3. solve8(M, rhs, h); return false if it fails.
//   4. Hn = [h0..h7, 1];  undo the normalization:  H = Td^-1 * Hn * Ts
//      (invertH from common.h and mat3Mul above); then divide all of H by H[8].
static bool fitHomography(const double* src, const double* dst, int n, double H[9]) {
    return false;
}

// TODO: count the inliers of H.
//   src, dst  all matched points (B side and A side), as x0, y0, x1, y1, ...
//   A pair is an inlier if applyH(H, src) lands within `thresh` pixels of dst
//   (compare squared distances: dx^2 + dy^2 < thresh^2).
//   If mask is not null, also set (*mask)[i] = 1 for inliers and 0 otherwise.
static int countInliers(const double H[9], const std::vector<double>& src, const std::vector<double>& dst, float thresh, std::vector<char>* mask) {
    return 0;
}

// TODO: RANSAC. Returns the number of inliers of the final H (0 = no homography found).
//   kA, kB   keypoints of A and B;  matches  from stage 4;  H  output
//   1. Build src (points in B: kB[m.b]) and dst (points in A: kA[m.a]) for every match.
//      H maps B -> A, so B is the source.
//   2. samples = makeRansacSamples(n, p.ransac_iters, p.ransac_seed)   (common.h)
//   3. For each hypothesis it: take its 4 matches, fitHomography (skip if it fails),
//      countInliers. Keep the H with the MOST inliers; on a tie keep the earlier one
//      (use '>' not '>=').
//   4. If the best count is < 4, return 0 (H stays the identity).
//   5. Refit: fitHomography on ALL inliers of the best H (least squares). Keep the refit if
//      it succeeds and has at least as many inliers. Return the final inlier count.
static int estimateHomography(const std::vector<Keypoint>& kA, const std::vector<Keypoint>& kB, const std::vector<Match>& matches, const Params& p, double H[9]) {
    return 0;
}

// =====================================================================================
// Stage 6 - warping + blending                                     owner: <Member B>
// =====================================================================================

// TODO: inverse warping + bilinear sampling + feathering.
//   A, B  the colour images;  H  maps B -> A;  out  the panorama
//   1. Canvas: start from A's rectangle [0, wA] x [0, hA] and extend it with B's four
//      corners mapped by H. Clamp the result to [-2 wA, 3 wA] x [-2 hA, 3 hA] so a bad H
//      cannot allocate a huge image. Offset (ox, oy) = (floor(minx), floor(miny));
//      size W = ceil(maxx) - ox, Hh = ceil(maxy) - oy. Fill out with black (0).
//   2. Hi = H^-1 (invertH), converted to float for the per-pixel maths.
//   3. For every output pixel (X, Y), with (px, py) = (X + ox, Y + oy) in A's frame:
//        - A: if (px, py) is inside A, take A's pixel; weight
//             wa = min(px + 1, wA - px, py + 1, hA - py)   (distance to A's border)
//        - B: (qx, qy) = Hi applied to (px, py). If 0 <= qx <= wB - 1 and 0 <= qy <= hB - 1:
//             bilinear sample of B at (qx, qy) from its 4 neighbours; weight
//             wb = min(qx + 1, wB - qx, qy + 1, hB - qy)
//        - out = (wa * colourA + wb * colourB) / (wa + wb), rounded (+0.5) and clamped to
//          255; leave black where neither image covers the pixel.
//      The weights fade each image out towards its own border, which hides the seam.
static void warpAndBlend(const ImageU8& A, const ImageU8& B, const double H[9], ImageU8& out) {
}

// =====================================================================================
// Pipeline (provided)
// =====================================================================================
// Runs the six stages in order and times each one with std::chrono.
// Stages 1-3 run once per image (A, then B); their times are summed.
// The ORB pattern and the Gaussian kernel are built inside the end-to-end time but outside
// the stage times.
void runV0(const ImageU8& A, const ImageU8& B, const Params& p, Result& result, Timing& t) {
    t = Timing();
    const Clock::time_point t_all = Clock::now();

    const std::vector<float> kernel = gaussianKernel(p.blur_sigma, p.blur_radius);
    const std::vector<OrbPair> pattern = makeOrbPattern(p.orb_seed, p.orb_radius);

    // ---- stages 1-3: each image independently (the GPU versions can overlap these)
    const ImageU8* images[2] = {&A, &B};
    std::vector<Keypoint> kps[2];
    std::vector<Descriptor> desc[2];
    for (int i = 0; i < 2; i++) {
        Clock::time_point t0 = Clock::now();
        ImageF gray, blurred;
        toGray(*images[i], gray);
        gaussianBlur(gray, blurred, kernel, p.blur_radius);
        t.stage_ms[ST_PREPROCESS] += msSince(t0);

        t0 = Clock::now();
        std::vector<float> R;
        harrisResponse(blurred, p, R);
        selectKeypoints(R, blurred.width, blurred.height, p, kps[i]);
        t.stage_ms[ST_DETECT] += msSince(t0);

        t0 = Clock::now();
        describe(blurred, kps[i], pattern, p.orb_radius, desc[i]);
        t.stage_ms[ST_DESCRIBE] += msSince(t0);
    }

    // ---- stage 4: A's descriptors against B's
    Clock::time_point t0 = Clock::now();
    std::vector<Match> matches;
    matchDescriptors(desc[0], desc[1], p, matches);
    t.stage_ms[ST_MATCH] = msSince(t0);

    // ---- stage 5: H maps B -> A
    t0 = Clock::now();
    result.num_inliers = estimateHomography(kps[0], kps[1], matches, p, result.H);
    t.stage_ms[ST_HOMOGRAPHY] = msSince(t0);

    // ---- stage 6
    t0 = Clock::now();
    warpAndBlend(A, B, result.H, result.panorama);
    t.stage_ms[ST_WARP] = msSince(t0);

    t.total_ms = msSince(t_all);
    result.num_kp_a = (int)kps[0].size();
    result.num_kp_b = (int)kps[1].size();
    result.num_matches = (int)matches.size();
    if (result.num_inliers == 0) fprintf(stderr, "warning: no homography found, panorama uses the identity\n");
}
