// =====================================================================================
// cuda_utils.cuh - error checking and CUDA-event timing shared by the GPU versions.
// Authors: <Member A>, <Member B>, <Member C>
// CHECK and GpuTimer adapted from the CSC14120 homework templates.
// =====================================================================================
//
// Include this from .cu files only (it needs the CUDA runtime).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call:   CHECK(cudaMalloc(&d_img, bytes));
// If the call fails, prints the file, line and CUDA's error message, then exits.
// Kernel launches return nothing, so after a launch check them with:
//   CHECK(cudaGetLastError());          // launch errors (bad configuration, ...)
//   CHECK(cudaDeviceSynchronize());     // errors while the kernel ran (debug builds only:
//                                       // it stops the CPU, so keep it out of timed code)
#define CHECK(call)                                                              \
    {                                                                            \
        const cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                              \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
            fprintf(stderr, "code: %d, reason: %s\n", error,                     \
                    cudaGetErrorString(error));                                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

// Measures GPU time with CUDA events, as the spec requires.
// Kernel launches return immediately, so a CPU clock around them measures nothing useful;
// events are recorded in the stream and timestamped by the GPU itself.
//
// Usage:
//   GpuTimer t;                    // or GpuTimer t(stream) to time work on a given stream
//   t.Start();
//   my_kernel<<<grid, block>>>(...);
//   t.Stop();
//   timing.stage_ms[ST_PREPROCESS] += t.Elapsed();   // waits for the kernel, returns ms
struct GpuTimer {
    cudaEvent_t start, stop;
    cudaStream_t stream;            // stream the events are recorded in (0 = default stream)

    explicit GpuTimer(cudaStream_t s = 0) : stream(s) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() { cudaEventRecord(start, stream); }
    void Stop() { cudaEventRecord(stop, stream); }

    // Blocks until the stop event has happened, then returns the time between the events.
    float Elapsed() {
        float ms;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Prints the GPU name, compute capability and SM count.
// The spec asks for the GPU name next to every result, so main.cu calls this for every
// version, including V0 (so the CPU and GPU numbers are tied to the same Colab session).
inline void printGpuName() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("GPU: none detected\n");
        return;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
}
