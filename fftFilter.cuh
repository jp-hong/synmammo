

#ifndef _FFT_FILTER
#define _FFT_FILTER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "util.h"

#define FILTER_OUT_LARGE_STRUCTURES true	//filter out structures larger than filter diameter
#define FILTER_OUT_SMALL_STRUCTURES false	//filter out structures smaller than filter diameter

//filtered image overwrites original image
void bandPassFilter(float *d_img, int x, int y, float filterLargeDiameter, float filterSmallDiameter, int stripesHorVert, float toleranceDiameter);
void bandPassFilter(float *d_img, int x, int y, float filterDiameter, bool largeOrSmall, int stripesHorVert, float toleranceDiameter);

//filtered image is stored in "d_result"
void bandPassFilter(float *d_img, float *d_result, int x, int y, float filterLargeDiameter, float filterSmallDiameter, int stripesHorVert, float toleranceDiameter);
void bandPassFilter(float *d_img, float *d_result, int x, int y, float filterDiameter, bool largeOrSmall, int stripesHorVert, float toleranceDiameter);

//CUDA kernel to assist in function "bandPassFilter()"
__global__ void bandPassFilterKernel(cufftComplex *fc, int len, float filterLarge, float filterSmall, int stripesHorVert, float scaleStripes);
__global__ void filterSmallStructuresKernel(cufftComplex *fc, int len, float filterSmall, int stripesHorVert, float scaleStripes);
__global__ void filterLargeStructuresKernel(cufftComplex *fc, int len, float filterLarge, int stripesHorVert, float scaleStripes);

__global__ void addZeroPadding(float *fPadded, float *f, int paddedLen, int x, int y);
__global__ void addZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y);

__global__ void removeZeroPadding(float *fPadded, float *f, int paddedLen, int x, int y);
__global__ void removeZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y);

__global__ void realToComplex(float *f, cufftComplex *fc, int len);
__global__ void complexToReal(cufftComplex *fc, float *f, int len);

#endif