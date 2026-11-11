// =====================================================================================
// versions.h - the interface every pipeline version implements, and the version table.
// Authors: Quách Châu Hạo Kiệt
// Drafted with an AI assistant (Claude) and reviewed by the team; see the AI declaration
// in the notebook.
// =====================================================================================
//
// The spec asks for ONE program that runs every version, selected with --version N, so all
// versions run on exactly the same input. This file is the contract that makes that work:
//
//   * every version is one function with the PipelineFn signature below;
//   * main.cu looks the requested version up in VERSIONS and calls it.
//
// Adding a new version (e.g. V2):
//   1. write  void runV2(...)  in its own file, e.g. src/v2_<idea>.cu
//   2. declare it below, next to runV0 / runV1
//   3. add a row  {2, "V2 <idea>", runV2}  to VERSIONS
//   4. add the new file to SRC in the Makefile and to the nvcc line in build.bat
#pragma once

#include "common.h"

// Runs the WHOLE pipeline (stages 1-6) once.
//   A       left image; the panorama is drawn in A's coordinate frame
//   B       right image; it is warped onto A
//   params  the shared parameters (see common.h)
//   result  output: panorama, homography H (B -> A), keypoint / match / inlier counts
//   timing  output: per-stage times, copy times and the end-to-end time of this run
// The function must not load or save files and must not print timing itself:
// main.cu does the warm-up, the repeated runs and the report.
typedef void (*PipelineFn)(const ImageU8& A, const ImageU8& B, const Params& params,
                           Result& result, Timing& timing);

void runV0(const ImageU8& A, const ImageU8& B, const Params& params, Result& result, Timing& timing);  // v0_sequential.cpp
void runV1(const ImageU8& A, const ImageU8& B, const Params& params, Result& result, Timing& timing);  // v1_basic.cu

// One row per version: the number given to --version, a display name, and the function.
struct VersionEntry {
    int id;
    const char* name;
    PipelineFn run;
};

static const VersionEntry VERSIONS[] = {
    {0, "V0 sequential", runV0},
    {1, "V1 basic CUDA", runV1},
};
static const int NUM_VERSIONS = sizeof(VERSIONS) / sizeof(VERSIONS[0]);
