// v1_basic.cu - V1: every stage on the GPU with straightforward kernels (global memory only).
// Authors: <Member A> (stages 1, 5), <Member B> (stages 2, 6), <Member C> (stages 3, 4)
//
// Not implemented yet: this placeholder keeps the program building for the midterm.
// Planned thread mapping (see the GPU plan in the notebook):
//   1. grayscale + blur      one thread per pixel (blur as two passes: rows, then columns)
//   2. Harris + NMS          one thread per pixel; compaction with a global atomic counter;
//                            top-K by sorting candidates on the device
//   3. ORB                   one thread per keypoint (orientation + 256 tests)
//   4. matching              one thread per keypoint of A scanning all of B (and B -> A for cross-check)
//   5. RANSAC                one thread per hypothesis (samples from makeRansacSamples),
//                            inlier count per thread, then argmax; refit on the host or a single block
//   6. warp + blend          one thread per output pixel
// Time each stage with GpuTimer (CUDA events) and the copies with h2d_ms / d2h_ms.
#include "versions.h"
#include "cuda_utils.cuh"

void runV1(const ImageU8& A, const ImageU8& B, const Params& params, Result& result, Timing& timing) {
    (void)A; (void)B; (void)params; (void)result; (void)timing;
    fprintf(stderr, "V1 is not implemented yet.\n");
    exit(EXIT_FAILURE);
}
