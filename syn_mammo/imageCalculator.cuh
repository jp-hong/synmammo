

#ifndef _IMAGE_CALCULATOR
#define _IMAGE_CALCULATOR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cmath"
#include "util.h"


//calculate projection and maximum intensity projection
__global__ void projectionAndMIP(float *projection, float *mip, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz);

__global__ void sinoproj(float *sinomat, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz);

__global__ void sinoproj_max(float *sinomat_max, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz);

//multiply "sinoMax" by "scaleFactor" then add to "sino" and store in "sinoResult"
__global__ void combineImages(const float *sino, const float *sinoMax, float *sinoResult, int len, float scaleFactor);

//add "offset" to all elements of array "f"
__global__ void offsetImage(float *f, int len, float offset);

float mean(float *f, int len);

__global__ void sumKernel(float *f, int len, float *globalSum);

float standardDeviation(float *f, int len);

__global__ void standardDeviationKernel(float *f, int len, float mean, float *globalSum);

__global__ void forceNonNegativity(float *f, int len);


#endif