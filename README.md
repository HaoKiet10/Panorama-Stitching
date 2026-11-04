# CSC14120 Final Project: Panorama Stitching (Group XX)

Stitches two overlapping photos into one panorama with a six-stage pipeline (grayscale + blur, Harris corners, ORB descriptors, Hamming matching, RANSAC homography, warping + feathering), in a sequential version (V0) and CUDA versions (V1+).

Presentation video: *(YouTube unlisted link, final submission)*

## Layout

```
GroupXX_Midterm.ipynb      midterm report (Colab)
Makefile                   Linux / Colab build
build.bat                  Windows build (CUDA Toolkit + Visual Studio Build Tools)
src/
  common.h                 shared types, parameters, ORB pattern, RANSAC samples, homography helpers
  versions.h               the run_vN interface and the version table
  cuda_utils.cuh           CHECK macro, CUDA-event GpuTimer, GPU name
  main.cu                  command line, image I/O, warm-up, repeated runs, timing report
  stb_impl.cpp             stb_image / stb_image_write implementation
  stb_image.h, stb_image_write.h
  v0_sequential.cpp        V0: single-threaded reference
  v1_basic.cu              V1: basic CUDA (placeholder until implemented)
tools/
  prepare_dataset.py       crop to 16:9 and resize to 1280x720 / 1920x1080 / 3840x2160
requirements.txt           Python packages for the tools and the notebook
dataset/<site>/<small|medium|large>/left.jpg, right.jpg
dataset/synthetic/<size>/left, right, H_gt.txt   synthetic pair from the starter notebook
```

## Requirements

- NVIDIA GPU and CUDA Toolkit 12 or newer (tested locally: RTX 5060 Laptop, CUDA 13.4; Colab: T4)
- A C++17 host compiler (g++ on Linux/Colab, Visual Studio 2022+ Build Tools on Windows)
- Python 3 with the packages in `requirements.txt` (preinstalled on Colab): `pip install -r requirements.txt`

## Build

Colab / Linux:

```bash
make
```

Pass `ARCH=-arch=sm_75` if `-arch=native` is not supported by your nvcc.

Windows:

```bat
build.bat
```

## Run

```bash
./stitch <left_image> <right_image> <output_image> --version <0|1|...> [--runs N]
```

| Option | Meaning |
|---|---|
| `--version N` | pipeline version (0 = sequential) |
| `--runs N` | timed runs after one warm-up run, averaged (default 5) |
| `--gt H.txt` | ground-truth homography (B to A); prints the corner error |
| `--ref-h H.txt` | reference homography, e.g. V0's, to check a GPU version; prints the corner error |
| `--save-h H.txt` | write the estimated homography |
| `--keypoints K`, `--ratio R`, `--iters N`, `--thresh PX` | override the main parameters |

Example:

```bash
./stitch dataset/site1/medium/left.jpg dataset/site1/medium/right.jpg out/site1.jpg --version 0 --runs 5
```

Expected output: a per-stage timing table, the keypoint/match/inlier counts, H, the panorama file, and a final `CSV,...` line that the notebook parses. With `--gt` on the synthetic pair it also prints the corner error, which should be under 2 px.

## Dataset

1. Shoot each pair by rotating the camera in place, with 30–50% overlap.
2. Put the photos in `raw/<site>/left.jpg` and `raw/<site>/right.jpg` (3 standard sites plus one folder whose name starts with `challeng` for the challenging pair).
3. Run `python tools/prepare_dataset.py raw dataset`.
4. Copy the starter notebook's synthetic pair into `dataset/synthetic/<small|medium|large>/` with its `H_gt.txt`.

## Midterm submission

Zip as `GroupXX_Midterm.zip`: `GroupXX_Midterm.ipynb` (run on Colab with all outputs saved) and `src/`. The notebook copies the team's shared Drive folder (`src/`, `tools/`, `Makefile`, `dataset/`) into the Colab session and builds from it.

Before submitting:
- [ ] Replace `<Member A/B/C>` in every source file header and the team table in the notebook
- [ ] Add the dataset link, site table and the written interpretation in section 3.5
- [ ] Set `DRIVE_DIR` and check that Runtime → Run all works on a fresh Colab GPU session
- [ ] Edit the AI usage declaration to match what the team did
