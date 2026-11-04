# Build the stitcher on Linux / Google Colab.
# Authors: <Member A>, <Member B>, <Member C>
#
#   make                       build ./stitch for the GPU in this machine
#   make ARCH=-arch=sm_75      build for a specific GPU (sm_75 = Colab's T4) if
#                              -arch=native is not supported by your nvcc
#   make clean                 delete the binary
#
# All versions are compiled into one program; --version picks one at run time.
# V0 is compiled with -O3 too, so the CPU baseline is a fair (optimized) one.
# When you add a version (e.g. src/v2_tiled.cu), add it to SRC.

NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS ?= -O3 -std=c++17

SRC = src/main.cu src/v0_sequential.cpp src/v1_basic.cu src/stb_impl.cpp
HDR = $(wildcard src/*.h src/*.cuh)

# Rebuild when any source or header changes.
stitch: $(SRC) $(HDR)
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $(SRC)

clean:
	rm -f stitch

.PHONY: clean
