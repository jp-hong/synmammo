

//standard C headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <math.h>

//CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

//user defined headers
#include "user.h"
#include "globals.h"
#include "util.h"
#include "imageCalculator.cuh"
#include "fftFilter.cuh"

using namespace std;

void loadImage(short *, size_t, int, int, int, char *);

void saveImage(float *, int, int, int, char *);

void saveImageSHRT(float *, int, int, int, char *);

void saveImageUSHRT(float *, int, int, int, char *);

void setFrameVectors(float *);

int main(){
	auto begin = chrono::high_resolution_clock::now();

	char *fileName = (char *)malloc(sizeof(char) * 256);

	dim3 GRID(0);
	dim3 BLOCK(256);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	//device memory pointers
	short *d_image1D = NULL;
	float *d_p = NULL;
	float *d_mip = NULL;
	float *d_p_filtered = NULL;
	float *d_mip_filtered = NULL;
	float *d_result = NULL;
	float *d_frameVectors = NULL;

	//set frame vectors
	HANDLE_ERROR(cudaMalloc((void**)&d_frameVectors, FRAME_BYTES));
	setFrameVectors(d_frameVectors);

	//read data
	HANDLE_ERROR(cudaMalloc((void **)&d_image1D, IMAGE_BYTES));
	HANDLE_ERROR(cudaMemset(d_image1D, 0, IMAGE_BYTES));
	generateFileName(fileName, INPUT_FILE_DIRECTORY, DATA_FILE_NAME);
	loadImage(d_image1D, IMAGE_BYTES, NX, NY, NZ, fileName);

	printf("read file\n");

	//maximum intensity projection of non-equalized data
	printf("Forward projection and maximum intensity projection\n");
	HANDLE_ERROR(cudaMalloc((void **)&d_p, DET_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&d_mip, DET_BYTES));
	HANDLE_ERROR(cudaMemset(d_p, 0, DET_BYTES));
	HANDLE_ERROR(cudaMemset(d_mip, 0, DET_BYTES));
	
	getGridDim(&GRID, BLOCK, DET_LEN);
	cudaEventRecord(start);
	projectionAndMIP <<<GRID, BLOCK>>> (d_p, d_mip, d_frameVectors, NU, NV, DU, DV, U0, V0, d_image1D, DX, DY, DZ, X0, Y0, Z0, NX, NY, NZ);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tRun time : %.1f ms\n", milliseconds);
	HANDLE_ERROR(cudaFree(d_image1D));

	//save projections
	if(SAVE_PROJECTIONS){
		generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, "proj", OUTPUT_FILE_EXTENSION);
		saveImage(d_p, NU, NV, DET_BYTES, fileName);

		generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, "max", OUTPUT_FILE_EXTENSION);
		saveImage(d_mip, NU, NV, DET_BYTES, fileName);
	}

	HANDLE_ERROR(cudaMalloc((void **)&d_p_filtered, DET_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&d_mip_filtered, DET_BYTES));
	HANDLE_ERROR(cudaMalloc((void **)&d_result, DET_BYTES));
	HANDLE_ERROR(cudaMemset(d_p_filtered, 0, DET_BYTES));
	HANDLE_ERROR(cudaMemset(d_mip_filtered, 0, DET_BYTES));
	HANDLE_ERROR(cudaMemset(d_result, 0, DET_BYTES));


	//bandpass filter
	printf("Filtering images\n");
	cudaEventRecord(start);
	bandPassFilter(d_mip, d_mip_filtered, NU, NV, FILTER_LARGE_DIAMETER, FILTER_OUT_LARGE_STRUCTURES, SUPPRESS_STRIPES, TOLERANCE_DIAMETER);
	bandPassFilter(d_p, d_p_filtered, NU, NV, FILTER_SMALL_DIAMETER, FILTER_OUT_SMALL_STRUCTURES, SUPPRESS_STRIPES, TOLERANCE_DIAMETER);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tRun time : %.1f ms\n", milliseconds);

	//save filtered images
	if(SAVE_FILTERED_IMAGES){
		generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, "proj_filtered", OUTPUT_FILE_EXTENSION);
		saveImage(d_p_filtered, NU, NV, DET_BYTES, fileName);

		generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, "max_filtered", OUTPUT_FILE_EXTENSION);
		saveImage(d_mip_filtered, NU, NV, DET_BYTES, fileName);
	}

	//determine scaling factor
	float p_min, p_max, mip_min, mip_max;
	float *h_img = (float *)malloc(DET_BYTES);
	
	HANDLE_ERROR(cudaMemcpy(h_img, d_p_filtered, DET_BYTES, cudaMemcpyDeviceToHost));

	p_min = h_img[0];
	p_max = h_img[0];

	for(int i = 1; i < DET_LEN; i++){
		p_min = MIN(p_min, h_img[i]);
		p_max = MAX(p_max, h_img[i]);
	}

	HANDLE_ERROR(cudaMemcpy(h_img, d_mip_filtered, DET_BYTES, cudaMemcpyDeviceToHost));

	mip_min = h_img[0];
	mip_max = h_img[0];

	for(int i = 1; i < DET_LEN; i++){
		mip_min = MIN(mip_min, h_img[i]);
		mip_max = MAX(mip_max, h_img[i]);
	}

	float factor = (p_max - p_min) / (mip_max - mip_min);
	printf("Scaling factor : %f\n", factor);

	free(h_img);

	//scale maximum intensity projection and add to projection
	printf("Combining images\n");
	getGridDim(&GRID, BLOCK, DET_LEN);
	cudaEventRecord(start);
	combineImages <<<GRID, BLOCK>>> (d_p_filtered, d_mip_filtered, d_result, DET_LEN, factor);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tRun time : %.1f ms\n", milliseconds);

	forceNonNegativity <<<GRID, BLOCK>>> (d_result, DET_LEN);

	//save result
	generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, "SM", OUTPUT_FILE_EXTENSION);
	//generateFileName(fileName, OUTPUT_FILE_DIRECTORY, OUTPUT_FILE_NAME, (int)factor, OUTPUT_FILE_EXTENSION);
	saveImageSHRT(d_result, NU, NV, DET_BYTES, fileName);

	//free remaining memory
	HANDLE_ERROR(cudaFree(d_frameVectors));
	HANDLE_ERROR(cudaFree(d_p));
	HANDLE_ERROR(cudaFree(d_mip));
	HANDLE_ERROR(cudaFree(d_p_filtered));
	HANDLE_ERROR(cudaFree(d_mip_filtered));
	HANDLE_ERROR(cudaFree(d_result));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	free(fileName);

	//device reset prior to exit
	HANDLE_ERROR(cudaDeviceReset());

	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = chrono::duration_cast<chrono::milliseconds>(dur).count();
	cout << "Total execution time : " << ms <<" ms" << endl;

	return 0;
}

void loadImage(short *d_image1D, size_t nBytes, int nx, int ny, int nz, char *fileName){
	short *h_image1D = (short*)malloc(nBytes);
	if (h_image1D == NULL)
	{
		printf("memory error\n");
		exit(EXIT_FAILURE);
	}
	readArrayFromFile((void*) h_image1D, nx * ny * nz, fileName, sizeof(short));
	HANDLE_ERROR(cudaMemcpy(d_image1D, h_image1D, nBytes, cudaMemcpyHostToDevice));
	free(h_image1D);
}

void saveImage(float *d_data1D, int nu, int nv, int nBytes, char *fileName){
	float *h_data1D = (float*)malloc(nBytes);
	HANDLE_ERROR(cudaMemcpy(h_data1D, d_data1D, nBytes, cudaMemcpyDeviceToHost));
	writeArrayToFile(h_data1D, nu * nv, fileName, sizeof(float));
	free(h_data1D);
	printf("Wrote file \"%s\"\n", fileName);
}

void saveImageSHRT(float *d_data1D, int nu, int nv, int nBytes, char *fileName){
	float *h_data1D = (float*)malloc(nBytes);
	short *h_data1D_shrt = (short *)malloc(sizeof(short) * nu * nv);
	HANDLE_ERROR(cudaMemcpy(h_data1D, d_data1D, nBytes, cudaMemcpyDeviceToHost));

	float max = h_data1D[0];

	for (int i = 1; i < nu * nv; i++)
		max = MAX(max, h_data1D[i]);

	for (int i = 0; i < nu * nv; i++)
		h_data1D_shrt[i] = h_data1D[i] * USHRT_MAX / max + SHRT_MIN;

	writeArrayToFile(h_data1D_shrt, nu * nv, fileName, sizeof(short));
	free(h_data1D);
	free(h_data1D_shrt);
	printf("Wrote file \"%s\"\n", fileName);
}

void saveImageUSHRT(float *d_data1D, int nu, int nv, int nBytes, char *fileName){
	float *h_data1D = (float*)malloc(nBytes);
	unsigned short *h_data1D_ushrt = (unsigned short *)malloc(sizeof(unsigned short) * nu * nv);
	HANDLE_ERROR(cudaMemcpy(h_data1D, d_data1D, nBytes, cudaMemcpyDeviceToHost));

	float max = h_data1D[0];

	for (int i = 1; i < nu * nv; i++)
		max = MAX(max, h_data1D[i]);

	for (int i = 0; i < nu * nv; i++)
		h_data1D_ushrt[i] = h_data1D[i] * USHRT_MAX / max;
	
	writeArrayToFile(h_data1D_ushrt, nu * nv, fileName, sizeof(unsigned short));
	free(h_data1D);
	free(h_data1D_ushrt);
	printf("Wrote file \"%s\"\n", fileName);
}

void setFrameVectors(float *d_frameVectors){
	float *h_frameVectors = (float*)malloc(FRAME_BYTES);

	h_frameVectors[0] = X0 + SOURCE_TO_DETECTOR;	//xSource
	h_frameVectors[1] = 0;							//ySource
	h_frameVectors[2] = 0;							//zSource

	h_frameVectors[3] = X0 - AXIS_TO_DETECTOR;		//xDetCenter
	h_frameVectors[4] = 0;							//yDetCenter
	h_frameVectors[5] = 0;							//zDetCenter

	h_frameVectors[6] = 0;							//eux
	h_frameVectors[7] = 1;							//euy
	h_frameVectors[8] = 0;							//euz

	h_frameVectors[9] = 0;							//evx
	h_frameVectors[10] = 0;							//evy
	h_frameVectors[11] = 1;							//evz

	HANDLE_ERROR(cudaMemcpy(d_frameVectors, h_frameVectors, FRAME_BYTES, cudaMemcpyHostToDevice));
	free(h_frameVectors);
}