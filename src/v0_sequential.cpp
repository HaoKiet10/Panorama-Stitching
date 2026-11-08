// =====================================================================================
// v0_sequential.cpp - V0: single-threaded reference implementation of the whole pipeline.
// Authors: <Member A> (stages 1, 5), Tôn Thất Minh Đăng (stages 2, 6), <Member C> (stages 3, 4)
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
// Stage 1 - preprocessing                                    
// Owner: Quách Châu Hạo Kiệt
// =====================================================================================

// TODO: convert an RGB image to grayscale.
//   in   RGB image, 3 bytes per pixel
//   out  float image of the same size
//   For every pixel: gray = 0.299 R + 0.587 G + 0.114 B   (values stay in 0..255)
static void toGray(const ImageU8& in, ImageF& out) {
    out.height = in.height;
    out.width = in.width;
    out.data.resize((size_t)out.width * out.height);
    
    for (int y = 0; y < out.height; y++) {
        for (int x = 0; x < out.width; x++) {
            size_t index = (size_t)y * out.width + x;
            size_t rgb_index = index * 3;
            out.data[index] = in.data[rgb_index] * .299f + in.data[rgb_index + 1] * .587f + in.data[rgb_index + 2] * .114f;
        }
    }
}

// TODO: build a normalized 1-D Gaussian kernel.
//   returns 2 * radius + 1 weights; weight i (i = -radius..radius) is proportional to
//   exp(-i^2 / (2 sigma^2)). Divide by the sum so the weights add up to 1
//   (otherwise the blur changes the image brightness).
static std::vector<float> gaussianKernel(float sigma, int radius) {
    std::vector<float> k(2 * radius + 1);
    float sum = 0;

    for (int i = -radius; i <= radius; i++) {
        k[i + radius] = expf(-(i * i) / (2 *  sigma * sigma));
        sum += k[i + radius];
    }

    for (int i = 0; i < k.size(); i++) {
        k[i] /= sum;
    }
    return k;
}

// TODO: separable Gaussian blur.
//   in, out  float images of the same size;  k  the kernel from gaussianKernel
//   1. horizontal pass into a temporary image:  tmp(x, y) = sum_i k[i] * in(x + i, y)
//   2. vertical pass into out:                  out(x, y) = sum_i k[i] * tmp(x, y + i)
//   with i = -radius..radius and clamp-to-edge at the borders (clampInt).
//   Two 1-D passes give the same result as one 2-D k x k blur with 2k instead of k^2 work.
static void gaussianBlur(const ImageF& in, ImageF& out, const std::vector<float>& k, int radius) {
    int height = in.height, width = in.width;
    std::vector<float> temp((size_t)height * width);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
                sum += k[i + radius] * in.data[y * width + clampInt(x + i, 0, width - 1)];
            temp[y * width + x] = sum;
        }
    }

    out.height = height;
    out.width = width;
    out.data.resize((size_t) height * width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
                sum += k[i + radius] * temp[clampInt(y + i, 0, height - 1) * width + x];
            out.data[y * width + x] = sum;
        }
    }
}

// =====================================================================================
// Stage 2 - keypoint detection                                     
// owner: Tôn Thất Minh Đăng
// =====================================================================================

// TODO: Harris corner response for every pixel.
//   img  the blurred grayscale image;  R  output, one float per pixel (resize it to w * h)
//   1. Sobel gradients at every pixel (clamp-to-edge):
//        gx = [I(x+1,y-1) + 2 I(x+1,y) + I(x+1,y+1)] - [I(x-1,y-1) + 2 I(x-1,y) + I(x-1,y+1)]
//        gy = [I(x-1,y+1) + 2 I(x,y+1) + I(x+1,y+1)] - [I(x-1,y-1) + 2 I(x,y-1) + I(x+1,y-1)]
//      and store the products gx*gx, gy*gy, gx*gy in three arrays.
//   2. For every pixel, sum each product over the (2 r + 1)^2 window, r = p.harris_radius
//      (clamp-to-edge):  Sxx, Syy, Sxy. Window function used: Gaussian with sigma =
//   3. R = (Sxx * Syy - Sxy^2) - p.harris_k * (Sxx + Syy)^2
//      R is large and positive at corners, negative on edges, near 0 in flat areas.
static void harrisResponse(const ImageF& img, const Params& p, std::vector<float>& R) {
    int w = img.width, h = img.height;
    size_t n = size_t(w * h);
    R.resize(n);

    // gradient maps
    ImageF Ixx{w, h, std::vector<float>(n)};
    ImageF Iyy{w, h, std::vector<float>(n)};
    ImageF Ixy{w, h, std::vector<float>(n)};

    //  1. Sobel gradients and second degree gradient calculation
    for(int r = 0; r < h; r++) {
        for(int c = 0; c < w; c++) {
            // sobel gradients
            float gx = 0, gy = 0;
            for(int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    int curRow = clampInt(r + i - 1, 0, h - 1);
                    int curCol = clampInt(c + j - 1, 0, w - 1);

                    int filterIdx = i * 3 + j;
                    int imgIdx = curRow * w + curCol;

                    gx += img.data[imgIdx] * p.sobelX[filterIdx];
                    gy += img.data[imgIdx] * p.sobelY[filterIdx];
                }
            }

            // second degree gradients
            int idx = r * w + c;
            Ixx.data[idx] = gx * gx;
            Iyy.data[idx] = gy * gy;
            Ixy.data[idx] = gx * gy;
        }
    }

    // calculate second moment matrix (Harris matrix) (Sxx, Sxy, Syy)
    // using 2 1D gaussian convolutions.
    float harris_sigma = p.harris_radius / 3.0f;
    std::vector<float> kernel = gaussianKernel(harris_sigma, p.harris_radius);
 
    ImageF Sxx, Syy, Sxy;
    gaussianBlur(Ixx, Sxx, kernel, p.harris_radius);
    gaussianBlur(Iyy, Syy, kernel, p.harris_radius);
    gaussianBlur(Ixy, Sxy, kernel, p.harris_radius);
    
    // From the second moment matrix find the determinant and trace values for each pixel
    // The two eigenvalues from this matrix (2*2) decide the direction of fastest and slowest intensity change
    // As gradient change the most when moving perpendicular to an edge => larger gradients
    // indicate a steeper edge in one direction.
    // As a corner requires steep changes in both directions at once => if both eigenvalues
    // are large, that pixel is a corner.
    for(int r = 0; r < h; r++) {
        for(int c = 0; c < w; c++) {
            int idx = r * w + c;
            float curSxx = Sxx.data[idx];
            float curSyy = Syy.data[idx];
            float curSxy = Sxy.data[idx];
            
            float det = curSxx * curSyy - curSxy * curSxy;
            float trace = curSxx + curSyy;

            R[idx] = det - p.harris_k * trace * trace;
        }
    }
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
    // run from border to ensure full 31 * 31 ORB zones later in stage 3.
    for(int r = p.border; r < h - p.border; r++) {
        for(int c = p.border; c < w - p.border; c++) {
            int idx = r * w + c;
            float score = R[idx];
            if(score <= 0.0f) continue; // only consider positive responses.
            // does not need thresholding because at the end we will sort by magnitude and
            // choose top-K anyway.
            
            // non-maximum suppression in a 3x3 window (nms_radius = 1)
            int nmsR = p.nms_radius;
            int nmsZoneSize = p.nms_radius * 2 + 1;
            bool isMaximum = true;
            for(int i = 0; i < nmsZoneSize; i++) {
                for(int j = 0; j < nmsZoneSize; j++) {
                    if(i == nmsR && j == nmsR) continue;

                    int curRow = clampInt(r + i - nmsR, 0, h - 1);
                    int curCol = clampInt(c + j - nmsR, 0, w - 1);
                    int zoneIdx = curRow * w + curCol;

                    if(score < R[zoneIdx]) {
                        isMaximum = false;
                        break;
                    }
                    
                    if(score == R[zoneIdx] && zoneIdx < idx) {
                        isMaximum = false;
                        break;
                    }
                }
                if(isMaximum == false) break;
            }

            // push candidate into keypoint array.
            if (isMaximum) kps.push_back({c, r, score, 0.0f});
        }
    }

    // Sort candidates based on score
    std::sort(kps.begin(), kps.end(), customKpComp);

    // cutoff at p.max_keypoints
    if((int)kps.size() > p.max_keypoints) {
        kps.erase(kps.begin() + p.max_keypoints, kps.end());
    }
}

// =====================================================================================
// Stage 3 - description (ORB)                                  
// Owner: Phan Thanh Bình
// =====================================================================================

// TODO: ORB orientation of the keypoint at (cx, cy).
//   Over all offsets (dx, dy) with dx^2 + dy^2 <= radius^2 (a disc, not a square):
//     m10 += dx * I(cx + dx, cy + dy)
//     m01 += dy * I(cx + dx, cy + dy)
//   return atan2f(m01, m10)   -> the direction from the keypoint to the patch's
//                                 "centre of brightness", in radians.
//   No clamping needed: p.border keeps the disc inside the image. Accumulate in float.
static float orientation(const ImageF& img, int cx, int cy, int radius) {
    const int w = img.width;
    const int r2 = radius * radius;
    float m10 = 0.0f, m01 = 0.0f;

    // Intensity moments over a disc (not a square) of the given radius.
    // p.border guarantees the disc stays inside the image, so no clamping is needed.
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy > r2) continue;
            const float v = img.data[(size_t)(cy + dy) * w + (cx + dx)];
            m10 += (float)dx * v;
            m01 += (float)dy * v;
        }
    }
    // Direction from the keypoint to the patch's "centre of brightness".
    return atan2f(m01, m10);
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
    const int w = img.width;
    const int numTests = (int)pattern.size();  // 256 binary tests -> 8 x 32-bit words

    desc.resize(kps.size());

    for (size_t k = 0; k < kps.size(); k++) {
        const int cx = (int)kps[k].x;
        const int cy = (int)kps[k].y;

        // 1. Orientation of the patch; store it back in the keypoint.
        const float angle = orientation(img, cx, cy, radius);
        kps[k].angle = angle;
        const float c = cosf(angle);
        const float s = sinf(angle);

        // All bits start at 0.
        for (int j = 0; j < 8; j++) desc[k].w[j] = 0u;

        // 2. Rotated BRIEF: compare two rotated offsets around the keypoint.
        for (int i = 0; i < numTests; i++) {
            const OrbPair& t = pattern[i];

            const int rx1 = roundToInt(c * (float)t.x1 - s * (float)t.y1);
            const int ry1 = roundToInt(s * (float)t.x1 + c * (float)t.y1);
            const int rx2 = roundToInt(c * (float)t.x2 - s * (float)t.y2);
            const int ry2 = roundToInt(s * (float)t.x2 + c * (float)t.y2);

            const float v1 = img.data[(size_t)(cy + ry1) * w + (cx + rx1)];
            const float v2 = img.data[(size_t)(cy + ry2) * w + (cx + rx2)];

            if (v1 < v2) desc[k].w[i >> 5] |= 1u << (i & 31);
        }
    }
}

// =====================================================================================
// Stage 4 - matching                                               
// Owner: Phan Thanh Bình
// =====================================================================================

// TODO: Hamming distance = number of differing bits = sum over the 8 words of
//   popcount32(a.w[k] ^ b.w[k]).
static inline int hamming(const Descriptor& a, const Descriptor& b) {
    int d = 0;
    for (int k = 0; k < 8; k++) d += popcount32(a.w[k] ^ b.w[k]);
    return d;
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
    matches.clear();
    const int nA = (int)dA.size();
    const int nB = (int)dB.size();
    if (nA == 0 || nB == 0) return;

    // 1. Cross-check table: for every b, the closest a (ties -> smallest a, hence strict '<').
    std::vector<int> best_of_b;
    if (p.cross_check) {
        best_of_b.assign(nB, -1);
        for (int b = 0; b < nB; b++) {
            int bestDist = INT_MAX;
            for (int a = 0; a < nA; a++) {
                const int d = hamming(dA[a], dB[b]);
                if (d < bestDist) {
                    bestDist = d;
                    best_of_b[b] = a;
                }
            }
        }
    }

    // 2. For every a: best and second-best distance over all of B.
    for (int a = 0; a < nA; a++) {
        int best = INT_MAX, second = INT_MAX, bestB = -1;
        for (int b = 0; b < nB; b++) {
            const int d = hamming(dA[a], dB[b]);
            if (d < best) {          // strict '<': ties keep the smallest b
                second = best;
                best = d;
                bestB = b;
            } else if (d < second) { // an equal distance also becomes the second best
                second = d;
            }
        }

        // 3. Ratio test (in floats); accepted automatically if B has a single point.
        const bool ratioOk = (nB == 1) || ((float)best < p.ratio * (float)second);
        // Cross-check: a must also be the best partner of bestB.
        const bool crossOk = !p.cross_check || (best_of_b[bestB] == a);

        if (ratioOk && crossOk) matches.push_back(Match{a, bestB, best});  // increasing a
    }
}

// =====================================================================================
// Stage 5 - homography (RANSAC + DLT)                              
// owner: Quách Châu Hạo Kiệt
// =====================================================================================

// TODO: Hartley normalization of n points (pts = x0, y0, x1, y1, ...).
//   Build T (3x3, row-major) that moves the points' centroid (cx, cy) to the origin and
//   scales them so their mean distance from it is sqrt(2):
//     s = sqrt(2) / mean distance       T = | s  0  -s cx |
//                                           | 0  s  -s cy |
//                                           | 0  0    1   |
//   Without it the DLT equations mix numbers ~1 and ~10^6 and lose precision.
static void normalizePoints(const double* pts, int n, double T[9]) {
    double cx = 0, cy = 0;

    for (int i = 0; i < n; i++) {
        cx += pts[2 * i] / n;
        cy += pts[2 * i + 1] / n;
    }

    double meanDist = 0;
    for (int i = 0; i < n; i++) {
        double dx = pts[2 * i] - cx;
        double dy = pts[2 * i + 1] - cy;
        meanDist += sqrt(dx * dx + dy * dy) / n;
    }

    // check if meanDist is 0 
    // if yes shrink factor = 1 -> solve8 reports failure
    double shrinkFactor = meanDist > 1e-12 ? sqrt(2) / meanDist : 1;

    T[0] = shrinkFactor;
    T[1] = 0;
    T[2] = -shrinkFactor * cx; 
    T[3] = 0;
    T[4] = shrinkFactor;
    T[5] = -shrinkFactor * cy;
    T[6] = 0;
    T[7] = 0;
    T[8] = 1;
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
    double Ts[9] = {0}, Td[9] = {0};
    normalizePoints(src, n, Ts);
    normalizePoints(dst, n, Td);

    double M[8][8] = {0};
    double rhs[8] = {0};
    for (int i = 0; i < n; i++) {
        double x = src[2 * i] * Ts[0] + Ts[2];
        double y = src[2 * i + 1] * Ts[4] + Ts[5];
        double u = dst[2 * i] * Td[0] + Td[2];
        double v = dst[2 * i + 1] * Td[4] + Td[5];

        double r1[] = {x, y, 1, 0, 0, 0, -u * x, -u * y};
        double r2[] = {0, 0, 0, x, y, 1, -v * x, -v * y};

        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 8; b++) {
                M[b][a] += r1[a] * r1[b]; 
                M[b][a] += r2[a] * r2[b];
            }
            rhs[a] += r1[a] * u; 
            rhs[a] += r2[a] * v;
        }
    }
    double h[8] = {0};
    if(!solve8(M, rhs, h)) return false;

    double hn[] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1};
    double TdInv[9] = {0};
    if (!invertH(Td, TdInv)) return false;

    double temp[9] = {0};
    mat3Mul(TdInv, hn, temp);
    mat3Mul(temp, Ts, H);
    
    for(int i = 0;  i < 9; i++)
        H[i] = H[i] / H[8];

    return true;
}

// TODO: count the inliers of H.
//   src, dst  all matched points (B side and A side), as x0, y0, x1, y1, ...
//   A pair is an inlier if applyH(H, src) lands within `thresh` pixels of dst
//   (compare squared distances: dx^2 + dy^2 < thresh^2).
//   If mask is not null, also set (*mask)[i] = 1 for inliers and 0 otherwise.
static int countInliers(const double H[9], const std::vector<double>& src, const std::vector<double>& dst, float thresh, std::vector<char>* mask) {
    int inliers = 0;
    float threshSq = thresh * thresh;
    int size = src.size() / 2;

    for (int i = 0; i < size; i++) {
        double ox = 0, oy = 0;
        applyH(H, src[2 * i], src[2 * i + 1], ox, oy);

        double dx = dst[2 * i] - ox;
        double dy = dst[2 * i + 1] - oy;
        bool inlier = (dx * dx + dy * dy) < threshSq;

        if (inlier)
            inliers++;
        
        if (mask)
            (*mask)[i] = inlier;
    }
    return inliers;
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
    std::vector<double> src, dst;
    src.resize(matches.size() * 2);
    dst.resize(matches.size() * 2);

    for (int i = 0; i < matches.size(); i++) {
        dst[2 * i] = kA[matches[i].a].x;
        dst[2 * i + 1] = kA[matches[i].a].y;
        src[2 * i] = kB[matches[i].b].x;
        src[2 * i + 1] = kB[matches[i].b].y;
    }

    std::vector<int> samples = makeRansacSamples(matches.size(), p.ransac_iters, p.ransac_seed);

    if (samples.size() <= 0) return 0;
    double hBest[9] = {0};
    int bestInliers = 0;

    for (int i = 0; i < p.ransac_iters; i++) {
        double src4[8], dst4[8];
        double hc[9] = {0};
        for (int j = 0; j < 4; j++) {
            int index = samples[4 * i + j];
            src4[2 * j] = kB[matches[index].b].x;
            src4[2 * j + 1] = kB[matches[index].b].y;
            dst4[2 * j] = kA[matches[index].a].x;
            dst4[2 * j + 1] = kA[matches[index].a].y;
        }
        if (!fitHomography(src4, dst4, 4, hc)) continue;

        int c = countInliers(hc, src, dst, p.ransac_thresh, nullptr);
        if (c > bestInliers) {
            bestInliers = c;
            std::copy(hc, hc + 9, hBest);
        }
    }

    if (bestInliers < 4) return 0;
    
    std::vector<char> mask(matches.size());
    bestInliers = countInliers(hBest, src, dst, p.ransac_thresh, &mask);

    std::vector<double> inSrc, inDst;
    for (int i = 0; i < matches.size(); i++) {
        if (!mask[i]) continue;
        inSrc.push_back(src[2 * i]);
        inSrc.push_back(src[2 * i + 1]);
        inDst.push_back(dst[2 * i]);
        inDst.push_back(dst[2 * i +  1]);
    }

    double hr[9] = {0};
    if (!fitHomography(inSrc.data(), inDst.data(), bestInliers, hr)) {
        std::copy(hBest, hBest + 9, H);
        return bestInliers;
    }
    int allInliers = countInliers(hr, src, dst, p.ransac_thresh, nullptr);
    if (allInliers >= bestInliers) {
        std::copy(hr, hr + 9, H);
        return allInliers;
    }
    std::copy(hBest, hBest + 9, H);
    return bestInliers;
}

// =====================================================================================
// Stage 6 - warping + blending                                     
// owner: Tôn Thất Minh Đăng
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
    int wA = A.width, hA = A.height, wB = B.width, hB = B.height;
    // ---------------------------------------
    // 1. Get canvas size to stitch A and B on
    // ---------------------------------------
    double coordsBx[4];
    double coordsBy[4];

    applyH(H, 0, 0, coordsBx[0], coordsBy[0]);
    applyH(H, B.width, 0, coordsBx[1], coordsBy[1]);
    applyH(H, 0, B.height, coordsBx[2], coordsBy[2]);
    applyH(H, B.width, B.height, coordsBx[3], coordsBy[3]);
    
    double minX = 0, minY = 0, maxX = A.width, maxY = A.height;
    for(int i = 0; i < 4; i++) {
        minX = std::min(minX, coordsBx[i]);
        minY = std::min(minY, coordsBy[i]);
        maxX = std::max(maxX, coordsBx[i]);
        maxY = std::max(maxY, coordsBy[i]);
    }
    
    // clamp the results
    minX = clampFloat(minX, -2.0 * wA, 3.0 * wA);
    maxX = clampFloat(maxX, -2.0 * wA, 3.0 * wA);
    minY = clampFloat(minY, -2.0 * hA, 3.0 * hA);
    maxY = clampFloat(maxY, -2.0 * hA, 3.0 * hA);
    
    // Compute integer canvas dimensions and offsets
    int ox = (int)floor(minX);
    int oy = (int)floor(minY);
    int canvasW  = (int)ceil(maxX) - ox;
    int canvasH = (int)ceil(maxY) - oy;

    // Initialize the output image canvas filled with 0 (black)
    out.width = canvasW;
    out.height = canvasH;
    out.channels = 3;
    out.data.assign((size_t)canvasW * canvasH * 3, 0);

    // -------------------------------------------
    // 2. Inverting H
    // -------------------------------------------
    double Hi[9];
    if (!invertH(H, Hi)) return;

    // -------------------------------------------
    // 3. Calculate canvas pixel values based on 
    // position in A's coordinate system.
    // -------------------------------------------
    for(int r = 0; r < canvasH; r++) {
        for(int c = 0; c < canvasW; c++) {
            // calculate corresponding image A coordinates
            int py = r + oy;
            int px = c + ox;

            // find position back in B
            double qx, qy;
            applyH(Hi, (double)px, (double)py, qx, qy);

            // blending the two images' color with each other
            // image A weights
            int colorA[3] = {0, 0, 0};
            float weightA = 0.0f;
            if(px >= 0 && px < wA && py >= 0 && py < hA) {
                // get color value in A
                int idxA = (py * wA + px) * 3;
                colorA[0] = A.data[idxA];
                colorA[1] = A.data[idxA + 1];
                colorA[2] = A.data[idxA + 2];
                
                // get weight
                weightA = edgeDistance((float)px, (float)py, (float)wA, (float)hA);
            }
            // image B weights
            float weightB = 0.0f;
            float colorB[3];
            if(qx >= 0.0f && qx < (float)wB && qy >= 0.0f && qy < (float)hB) {
                // bilinear interpolation to get color value in B
                // get 4 pixels coords around q.
                int x0 = (int)floorf(qx), y0 = (int)floorf(qy);
                int x1 = x0 + 1, y1 = y0 + 1;

                float dx = qx - x0, dy = qy - y0;
                // clamp to image B bounds
                x0 = clampInt(x0, 0, wB - 1);
                x1 = clampInt(x1, 0, wB - 1);
                y0 = clampInt(y0, 0, hB - 1);
                y1 = clampInt(y1, 0, hB - 1);

                // sample value of 4 neighbor pixels to get final 
                // color value of B.
                for(int ch = 0; ch < 3; ch++) {
                    float p00 = B.data[(y0 * wB + x0) * 3 + ch];
                    float p01 = B.data[(y0 * wB + x1) * 3 + ch];
                    float p10 = B.data[(y1 * wB + x0) * 3 + ch];
                    float p11 = B.data[(y1 * wB + x1) * 3 + ch];

                    colorB[ch] = (1.0f - dy) * ((1.0f - dx) * p00 + dx * p01) + dy * ((1.0f - dx) * p10 + dx * p11);
                }

                // get B weight
                weightB = edgeDistance(qx, qy, (float)wB, (float)hB);
            }

            // blending
            int outIdx = (r * canvasW + c) * 3;
            if (weightA > 0.0f && weightB > 0.0f) {
                // Overlapping region: weighted avg (feathering)
                float sumW = weightA + weightB;
                for(int ch = 0; ch < 3; ch++) {
                    float blended = (weightA * colorA[ch] + weightB * colorB[ch]) / sumW;
                    out.data[outIdx + ch] = (uint8_t)clampInt(roundToInt(blended),0, 255);
                }
            }
            else if(weightA > 0.0f) {
                // Only Image A cover this pixel
                for(int ch = 0; ch < 3; ch++) {
                    out.data[outIdx + ch] = colorA[ch];
                }
            }
            else if(weightB > 0.0f) {
                // Only Image B cover this pixel
                for(int ch = 0; ch < 3; ch++) {
                    out.data[outIdx + ch] = (uint8_t)clampInt(roundToInt(colorB[ch]), 0, 255);
                }
            }
            else {
                // Neither image cover this pixel, leave black
                out.data[outIdx + 0] = 0;
                out.data[outIdx + 1] = 0;
                out.data[outIdx + 2] = 0;
            }
        }
    }
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