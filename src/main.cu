// =====================================================================================
// main.cu - command line, image I/O, warm-up, repeated runs and the timing report.
// Responsible: Tôn Thất Minh Đăng
// Drafted with an AI assistant (Claude) and reviewed by the team; see the AI declaration
// in the notebook.
// =====================================================================================
//
// Usage (the command line required by the spec, plus a few optional extras):
//   ./stitch <left_image> <right_image> <output_image> --version <0|1|...> [--runs N]
//            [--gt H.txt] [--ref-h H.txt] [--save-h H.txt]
//            [--keypoints K] [--ratio R] [--iters N] [--thresh PX]
//
// What main does, in order:
//   1. parse the arguments and pick the version from VERSIONS (versions.h)
//   2. load both images (not timed)
//   3. print the GPU name (the spec wants it next to every result)
//   4. run the pipeline once as a warm-up (excludes CUDA context creation and first-launch
//      overhead from the timings)
//   5. run it --runs times (default 5), average every time, keep the minimum end-to-end
//   6. print the timing table, counts, H, and the corner errors if --gt / --ref-h were given
//   7. save the panorama (not timed) and print one machine-readable "CSV," line
//
// The CSV line is what the notebook parses; its columns are listed just before it is printed.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

#include "stb_image.h"
#include "stb_image_write.h"
#include "versions.h"
#include "cuda_utils.cuh"

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <left_image> <right_image> <output_image> --version <0|1|...> [--runs N]\n"
            "          [--gt H.txt]      ground-truth homography (B -> A), prints the corner error\n"
            "          [--ref-h H.txt]   reference homography (e.g. V0's), prints the corner error\n"
            "          [--save-h H.txt]  write the estimated homography\n"
            "          [--keypoints K] [--ratio R] [--iters N] [--thresh PX]\n",
            prog);
    exit(EXIT_FAILURE);
}

// ------------------------------------------------------------------ file helpers

// Loads any format stb_image understands (JPG, PNG, BMP, ...) and forces 3 channels (RGB),
// so grayscale or RGBA files also work.
static bool loadImage(const char* path, ImageU8& img) {
    int w, h, c;
    uint8_t* pixels = stbi_load(path, &w, &h, &c, 3);
    if (!pixels) return false;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(pixels, pixels + (size_t)w * h * 3);
    stbi_image_free(pixels);
    return true;
}

// Saves as JPEG (quality 95) if the name ends in .jpg / .jpeg, otherwise as PNG.
static bool saveImage(const char* path, const ImageU8& img) {
    if (img.width <= 0 || img.height <= 0) return false;
    std::string p(path);
    std::string ext = p.size() >= 4 ? p.substr(p.size() - 4) : "";
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == "jpeg")
        return stbi_write_jpg(path, img.width, img.height, 3, img.data.data(), 95) != 0;
    return stbi_write_png(path, img.width, img.height, 3, img.data.data(), img.width * 3) != 0;
}

// Reads a homography as 9 numbers (row by row, any whitespace).
static bool loadH(const char* path, double H[9]) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    bool ok = true;
    for (int i = 0; i < 9 && ok; i++) ok = fscanf(f, "%lf", &H[i]) == 1;
    fclose(f);
    return ok;
}

// Writes a homography in the same format loadH reads (3 rows of 3 numbers).
static bool saveH(const char* path, const double H[9]) {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    for (int r = 0; r < 3; r++) fprintf(f, "%.10g %.10g %.10g\n", H[r * 3], H[r * 3 + 1], H[r * 3 + 2]);
    fclose(f);
    return true;
}

// ------------------------------------------------------------------ main

int main(int argc, char** argv) {
    // ---- 1. arguments
    if (argc < 4) usage(argv[0]);
    const char* left_path = argv[1];
    const char* right_path = argv[2];
    const char* out_path = argv[3];
    int version = 0, runs = 5;
    const char *gt_path = nullptr, *ref_path = nullptr, *save_h_path = nullptr;
    Params params;  // defaults from common.h; a few can be overridden below

    for (int i = 4; i < argc; i++) {
        // `need` returns the value after the current flag, or prints the usage if it is missing
        auto need = [&](void) -> const char* { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
        if (!strcmp(argv[i], "--version")) version = atoi(need());
        else if (!strcmp(argv[i], "--runs")) runs = std::max(1, atoi(need()));
        else if (!strcmp(argv[i], "--gt")) gt_path = need();
        else if (!strcmp(argv[i], "--ref-h")) ref_path = need();
        else if (!strcmp(argv[i], "--save-h")) save_h_path = need();
        else if (!strcmp(argv[i], "--keypoints")) params.max_keypoints = atoi(need());
        else if (!strcmp(argv[i], "--ratio")) params.ratio = (float)atof(need());
        else if (!strcmp(argv[i], "--iters")) params.ransac_iters = atoi(need());
        else if (!strcmp(argv[i], "--thresh")) params.ransac_thresh = (float)atof(need());
        else usage(argv[0]);
    }

    const VersionEntry* entry = nullptr;
    for (int i = 0; i < NUM_VERSIONS; i++)
        if (VERSIONS[i].id == version) entry = &VERSIONS[i];
    if (!entry) {
        fprintf(stderr, "Unknown version %d\n", version);
        return EXIT_FAILURE;
    }

    // ---- 2. load the images (not timed)
    ImageU8 A, B;
    if (!loadImage(left_path, A)) { fprintf(stderr, "Cannot read %s\n", left_path); return EXIT_FAILURE; }
    if (!loadImage(right_path, B)) { fprintf(stderr, "Cannot read %s\n", right_path); return EXIT_FAILURE; }

    // ---- 3. header
    printf("%s | A %dx%d | B %dx%d | runs %d\n", entry->name, A.width, A.height, B.width, B.height, runs);
    printGpuName();

    // ---- 4. warm-up run (its timing is thrown away)
    Result result;
    Timing timing;
    entry->run(A, B, params, result, timing);

    // ---- 5. timed runs: average every field, remember the fastest end-to-end time
    Timing avg;
    double min_total = 1e30;
    for (int r = 0; r < runs; r++) {
        Timing t;
        entry->run(A, B, params, result, t);  // `result` keeps the output of the last run
        for (int s = 0; s < NUM_STAGES; s++) avg.stage_ms[s] += t.stage_ms[s] / runs;
        avg.h2d_ms += t.h2d_ms / runs;
        avg.d2h_ms += t.d2h_ms / runs;
        avg.total_ms += t.total_ms / runs;
        min_total = std::min(min_total, t.total_ms);
    }

    // ---- 6. report
    // "share" = stage time / end-to-end time. For GPU versions, "stages total" is the total
    // kernel time and the copies are listed separately, as the spec asks.
    double stage_sum = 0;
    for (int s = 0; s < NUM_STAGES; s++) stage_sum += avg.stage_ms[s];
    printf("\n%-16s %10s %7s\n", "stage", "avg ms", "share");
    for (int s = 0; s < NUM_STAGES; s++)
        printf("%-16s %10.3f %6.1f%%\n", STAGE_NAMES[s], avg.stage_ms[s],
               avg.total_ms > 0 ? 100.0 * avg.stage_ms[s] / avg.total_ms : 0.0);
    printf("%-16s %10.3f\n", "stages total", stage_sum);
    printf("%-16s %10.3f\n", "host->device", avg.h2d_ms);
    printf("%-16s %10.3f\n", "device->host", avg.d2h_ms);
    printf("%-16s %10.3f   (min %.3f)\n", "end to end", avg.total_ms, min_total);

    printf("\nkeypoints A %d, B %d | matches %d | inliers %d\n", result.num_kp_a, result.num_kp_b,
           result.num_matches, result.num_inliers);
    printf("H (B -> A):\n");
    for (int r = 0; r < 3; r++)
        printf("  %12.6f %12.6f %12.6f\n", result.H[r * 3], result.H[r * 3 + 1], result.H[r * 3 + 2]);

    // Corner errors: -1 in the CSV means "not requested".
    double gt_err = -1, ref_err = -1;
    double G[9];
    if (gt_path) {
        if (!loadH(gt_path, G)) { fprintf(stderr, "Cannot read %s\n", gt_path); return EXIT_FAILURE; }
        gt_err = cornerError(result.H, G, B.width, B.height);
        printf("corner error vs ground truth: %.3f px\n", gt_err);
    }
    if (ref_path) {
        if (!loadH(ref_path, G)) { fprintf(stderr, "Cannot read %s\n", ref_path); return EXIT_FAILURE; }
        ref_err = cornerError(result.H, G, B.width, B.height);
        printf("corner error vs reference H: %.3f px\n", ref_err);
    }

    // ---- 7. save outputs (not timed)
    if (save_h_path && !saveH(save_h_path, result.H)) {
        fprintf(stderr, "Cannot write %s\n", save_h_path);
        return EXIT_FAILURE;
    }
    if (!saveImage(out_path, result.panorama)) {
        fprintf(stderr, "Cannot write %s (panorama is %dx%d)\n", out_path, result.panorama.width,
                result.panorama.height);
        return EXIT_FAILURE;
    }
    printf("panorama %dx%d saved to %s\n", result.panorama.width, result.panorama.height, out_path);

    // CSV columns (keep in sync with CSV_COLS in the notebook):
    // version, width, height, runs, 6 stage times, h2d, d2h, total, min_total,
    // kp_a, kp_b, matches, inliers, gt_err, ref_err
    printf("CSV,%d,%d,%d,%d", version, A.width, A.height, runs);
    for (int s = 0; s < NUM_STAGES; s++) printf(",%.4f", avg.stage_ms[s]);
    printf(",%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f,%.4f\n", avg.h2d_ms, avg.d2h_ms, avg.total_ms, min_total,
           result.num_kp_a, result.num_kp_b, result.num_matches, result.num_inliers, gt_err, ref_err);
    return 0;
}
