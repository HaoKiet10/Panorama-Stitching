// =====================================================================================
// stb_impl.cpp - compiles the stb image loader and writer once for the whole program.
// Authors: <Member A>, <Member B>, <Member C>
// Uses stb_image.h v2.30 and stb_image_write.h v1.16 by Sean Barrett (public domain),
// https://github.com/nothings/stb
// =====================================================================================
//
// stb libraries are "single-header": every file may #include the header to get the function
// declarations, but exactly ONE .cpp file must define the *_IMPLEMENTATION macro before
// including it, so the function bodies are compiled once. This is that file.
// main.cu includes the same headers without the macros, for the declarations only.
#define _CRT_SECURE_NO_WARNINGS  // silences MSVC warnings about sprintf/fopen inside stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
