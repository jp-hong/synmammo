

#include "fftFilter.cuh"


void bandPassFilter(float *d_img, int x, int y, float filterLargeDiameter, float filterSmallDiameter, int stripesHorVert, float toleranceDiameter){
	cufftHandle plan;
	cufftComplex *d_imgComplex;
	float *d_imgPadded;
	int IMG_LEN = x * y;
	int X_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * x - 1.0f)/logf(2.0f)))));
	int Y_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * y - 1.0f)/logf(2.0f)))));
	int PAD_LEN = MAX(X_PAD, Y_PAD);
	int FFT_LEN = PAD_LEN * PAD_LEN;
	int FFT_BYTES = FFT_LEN * sizeof(float);
	int FFT_COMPLEX_BYTES = FFT_LEN * sizeof(cufftComplex);

	dim3 BLOCK(256);
	dim3 GRID(0);

	HANDLE_ERROR(cudaMalloc((void**)&d_imgPadded, FFT_BYTES));
	HANDLE_ERROR(cudaMemset(d_imgPadded, 0, FFT_BYTES));
	HANDLE_ERROR(cudaMalloc((void**)&d_imgComplex, FFT_COMPLEX_BYTES));

	if(cufftPlan2d(&plan, PAD_LEN, PAD_LEN, CUFFT_C2C) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(EXIT_FAILURE);
	}

	getGridDim(&GRID, BLOCK, IMG_LEN);
	addZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	getGridDim(&GRID, BLOCK, FFT_LEN);
	realToComplex <<<GRID, BLOCK>>> (d_imgPadded, d_imgComplex, FFT_LEN);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	float filterLarge = 2.0f * filterLargeDiameter / (float)PAD_LEN;
	float filterSmall = 2.0f * filterSmallDiameter / (float)PAD_LEN;
	float sharpness = (100.0f - toleranceDiameter) / 100.0f;

	getGridDim(&GRID, BLOCK, FFT_LEN);
	bandPassFilterKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filterLarge, filterSmall, stripesHorVert, sharpness);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	complexToReal <<<GRID, BLOCK>>> (d_imgComplex, d_imgPadded, FFT_LEN);
	removeZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	cufftDestroy(plan);
	HANDLE_ERROR(cudaFree(d_imgPadded));
	HANDLE_ERROR(cudaFree(d_imgComplex));
}

void bandPassFilter(float *d_img, float *d_result, int x, int y, float filterLargeDiameter, float filterSmallDiameter, int stripesHorVert, float toleranceDiameter){
	cufftHandle plan;
	cufftComplex *d_imgComplex;
	float *d_imgPadded;
	int IMG_LEN = x * y;
	int X_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * x - 1.0f)/logf(2.0f)))));
	int Y_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * y - 1.0f)/logf(2.0f)))));
	int PAD_LEN = MAX(X_PAD, Y_PAD);
	int FFT_LEN = PAD_LEN * PAD_LEN;
	int FFT_BYTES = FFT_LEN * sizeof(float);
	int FFT_COMPLEX_BYTES = FFT_LEN * sizeof(cufftComplex);

	dim3 BLOCK(256);
	dim3 GRID(0);

	HANDLE_ERROR(cudaMalloc((void**)&d_imgPadded, FFT_BYTES));
	HANDLE_ERROR(cudaMemset(d_imgPadded, 0, FFT_BYTES));
	HANDLE_ERROR(cudaMalloc((void**)&d_imgComplex, FFT_COMPLEX_BYTES));

	if(cufftPlan2d(&plan, PAD_LEN, PAD_LEN, CUFFT_C2C) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}

	getGridDim(&GRID, BLOCK, IMG_LEN);
	addZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	getGridDim(&GRID, BLOCK, FFT_LEN);
	realToComplex <<<GRID, BLOCK>>> (d_imgPadded, d_imgComplex, FFT_LEN);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	float filterLarge = 2.0f * filterLargeDiameter / (float)PAD_LEN;
	float filterSmall = 2.0f * filterSmallDiameter / (float)PAD_LEN;
	float sharpness = (100.0f - toleranceDiameter) / 100.0f;

	getGridDim(&GRID, BLOCK, FFT_LEN);
	bandPassFilterKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filterLarge, filterSmall, stripesHorVert, sharpness);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
		return;
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	complexToReal <<<GRID, BLOCK>>> (d_imgComplex, d_imgPadded, FFT_LEN);
	removeZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_result, PAD_LEN, x, y);

	cufftDestroy(plan);
	HANDLE_ERROR(cudaFree(d_imgPadded));
	HANDLE_ERROR(cudaFree(d_imgComplex));
}

void bandPassFilter(float *d_img, int x, int y, float filterDiameter, bool largeOrSmall, int stripesHorVert, float toleranceDiameter){
	cufftHandle plan;
	cufftComplex *d_imgComplex;
	float *d_imgPadded;
	int IMG_LEN = x * y;
	int X_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * x - 1.0f)/logf(2.0f)))));
	int Y_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * y - 1.0f)/logf(2.0f)))));
	int PAD_LEN = MAX(X_PAD, Y_PAD);
	int FFT_LEN = PAD_LEN * PAD_LEN;
	int FFT_BYTES = FFT_LEN * sizeof(float);
	int FFT_COMPLEX_BYTES = FFT_LEN * sizeof(cufftComplex);

	dim3 BLOCK(256);
	dim3 GRID(0);

	HANDLE_ERROR(cudaMalloc((void**)&d_imgPadded, FFT_BYTES));
	HANDLE_ERROR(cudaMemset(d_imgPadded, 0, FFT_BYTES));
	HANDLE_ERROR(cudaMalloc((void**)&d_imgComplex, FFT_COMPLEX_BYTES));

	if(cufftPlan2d(&plan, PAD_LEN, PAD_LEN, CUFFT_C2C) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(EXIT_FAILURE);
	}

	getGridDim(&GRID, BLOCK, IMG_LEN);
	addZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	getGridDim(&GRID, BLOCK, FFT_LEN);
	realToComplex <<<GRID, BLOCK>>> (d_imgPadded, d_imgComplex, FFT_LEN);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	float filter = 2.0f * filterDiameter / (float)PAD_LEN;
	float sharpness = (100.0f - toleranceDiameter) / 100.0f;

	getGridDim(&GRID, BLOCK, FFT_LEN);

	if(largeOrSmall)
		filterLargeStructuresKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filter, stripesHorVert, sharpness);
	else
		filterSmallStructuresKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filter, stripesHorVert, sharpness);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	complexToReal <<<GRID, BLOCK>>> (d_imgComplex, d_imgPadded, FFT_LEN);
	removeZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	cufftDestroy(plan);
	HANDLE_ERROR(cudaFree(d_imgPadded));
	HANDLE_ERROR(cudaFree(d_imgComplex));
}

void bandPassFilter(float *d_img, float *d_result, int x, int y, float filterDiameter, bool largeOrSmall, int stripesHorVert, float toleranceDiameter){
	cufftHandle plan;
	cufftComplex *d_imgComplex;
	float *d_imgPadded;
	int IMG_LEN = x * y;
	int X_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * x - 1.0f)/logf(2.0f)))));
	int Y_PAD = (int)(powf(2.0f, (ceilf(logf(2.0f * y - 1.0f)/logf(2.0f)))));
	int PAD_LEN = MAX(X_PAD, Y_PAD);
	int FFT_LEN = PAD_LEN * PAD_LEN;
	int FFT_BYTES = FFT_LEN * sizeof(float);
	int FFT_COMPLEX_BYTES = FFT_LEN * sizeof(cufftComplex);

	dim3 BLOCK(256);
	dim3 GRID(0);

	HANDLE_ERROR(cudaMalloc((void**)&d_imgPadded, FFT_BYTES));
	HANDLE_ERROR(cudaMemset(d_imgPadded, 0, FFT_BYTES));
	HANDLE_ERROR(cudaMalloc((void**)&d_imgComplex, FFT_COMPLEX_BYTES));

	if(cufftPlan2d(&plan, PAD_LEN, PAD_LEN, CUFFT_C2C) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(EXIT_FAILURE);
	}

	getGridDim(&GRID, BLOCK, IMG_LEN);
	addZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_img, PAD_LEN, x, y);

	getGridDim(&GRID, BLOCK, FFT_LEN);
	realToComplex <<<GRID, BLOCK>>> (d_imgPadded, d_imgComplex, FFT_LEN);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	float filter = 2.0f * filterDiameter / (float)PAD_LEN;
	float sharpness = (100.0f - toleranceDiameter) / 100.0f;

	getGridDim(&GRID, BLOCK, FFT_LEN);
	
	if(largeOrSmall)
		filterLargeStructuresKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filter, stripesHorVert, sharpness);
	else
		filterSmallStructuresKernel <<<GRID, BLOCK>>> (d_imgComplex, PAD_LEN, filter, stripesHorVert, sharpness);

	if(cufftExecC2C(plan, d_imgComplex, d_imgComplex, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	complexToReal <<<GRID, BLOCK>>> (d_imgComplex, d_imgPadded, FFT_LEN);
	removeZeroPadding2D <<<GRID, BLOCK>>> (d_imgPadded, d_result, PAD_LEN, x, y);

	cufftDestroy(plan);
	HANDLE_ERROR(cudaFree(d_imgPadded));
	HANDLE_ERROR(cudaFree(d_imgComplex));
}

__global__ void bandPassFilterKernel(cufftComplex *fc, int len, float filterLarge, float filterSmall, int stripesHorVert, float scaleStripes){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len * len)
		return;

	int xIdx = idx % len;
	int yIdx = idx / len;

	if(xIdx > len / 2)
		xIdx = len - 1 - xIdx;

	if(yIdx > len / 2)
		yIdx = len - 1 - yIdx;

	float scaleLarge = filterLarge * filterLarge;
	float scaleSmall = filterSmall * filterSmall;
	scaleStripes *= scaleStripes;

	float rowFactLarge = expf(-(yIdx * yIdx) * scaleLarge);
	float rowFactSmall = expf(-(yIdx * yIdx) * scaleSmall);
	float colFactLarge = expf(-(xIdx * xIdx) * scaleLarge);
	float colFactSmall = expf(-(xIdx * xIdx) * scaleSmall);

	float factor = (1.0f - rowFactLarge * colFactLarge) * rowFactSmall * colFactSmall;

	switch(stripesHorVert){
	case 1: factor *= (1.0f - expf(-(xIdx * xIdx) * scaleStripes)); break;
	case 2: factor *= (1.0f - expf(-(yIdx * yIdx) * scaleStripes));
	}

	fc[idx].x *= factor;
	fc[idx].y *= factor;
}

__global__ void filterSmallStructuresKernel(cufftComplex *fc, int len, float filterSmall, int stripesHorVert, float scaleStripes){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len * len)
		return;

	int xIdx = idx % len;
	int yIdx = idx / len;

	if(xIdx > len / 2)
		xIdx = len - 1 - xIdx;

	if(yIdx > len / 2)
		yIdx = len - 1 - yIdx;

	float scaleSmall = filterSmall * filterSmall;
	scaleStripes *= scaleStripes;

	float rowFactSmall = expf(-(yIdx * yIdx) * scaleSmall);
	float colFactSmall = expf(-(xIdx * xIdx) * scaleSmall);

	float factor = rowFactSmall * colFactSmall;

	switch(stripesHorVert){
	case 1: factor *= (1.0f - expf(-(xIdx * xIdx) * scaleStripes)); break;
	case 2: factor *= (1.0f - expf(-(yIdx * yIdx) * scaleStripes));
	}

	fc[idx].x *= factor;
	fc[idx].y *= factor;
}

__global__ void filterLargeStructuresKernel(cufftComplex *fc, int len, float filterLarge, int stripesHorVert, float scaleStripes){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len * len)
		return;

	int xIdx = idx % len;
	int yIdx = idx / len;

	if(xIdx > len / 2)
		xIdx = len - 1 - xIdx;

	if(yIdx > len / 2)
		yIdx = len - 1 - yIdx;

	float scaleLarge = filterLarge * filterLarge;
	scaleStripes *= scaleStripes;

	float rowFactLarge = expf(-(yIdx * yIdx) * scaleLarge);
	float colFactLarge = expf(-(xIdx * xIdx) * scaleLarge);

	float factor = (1.0f - rowFactLarge * colFactLarge);

	switch(stripesHorVert){
	case 1: factor *= (1.0f - expf(-(xIdx * xIdx) * scaleStripes)); break;
	case 2: factor *= (1.0f - expf(-(yIdx * yIdx) * scaleStripes));
	}

	fc[idx].x *= factor;
	fc[idx].y *= factor;
}

__global__ void addZeroPadding(float *fPadded, float *f, int paddedLen, int x, int y){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= x * y)
		return;

	int padding = (paddedLen - x) / 2;
	int u = idx % x;
	int v = idx / x;

	fPadded[(u + padding) + paddedLen * v] = f[idx];
}

//__global__ void addZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y){
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if(idx >= x * y)
//		return;
//
//	int u_padding = (paddedLen - x) / 2;
//	int v_padding = (paddedLen - y) / 2;
//	int u = idx % x;
//	int v = idx / x;
//
//	fPadded[(u + u_padding) + paddedLen * (v + v_padding)] = f[idx];
//}

__global__ void addZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= x * y)
		return;

	int u_padding = (paddedLen - x) / 2;
	int v_padding = (paddedLen - y) / 2;
	int u = idx % x;
	int v = idx / x;

	fPadded[(u + u_padding) + paddedLen * (v + v_padding)] = f[idx];
}

__global__ void removeZeroPadding(float *fPadded, float *f, int paddedLen, int x, int y){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= paddedLen * y)
		return;

	int padding = (paddedLen - x) / 2;
	int u = idx % paddedLen - padding;
	int v = idx / paddedLen;

	if(u < 0 || u >= x)
		return;

	f[u + x * v] = fPadded[idx];
}

//__global__ void removeZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y){
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if(idx >= paddedLen * paddedLen)
//		return;
//
//	int u_padding = (paddedLen - x) / 2;
//	int v_padding = (paddedLen - y) / 2;
//	int u = idx % paddedLen - u_padding;
//	int v = idx / paddedLen - v_padding;
//
//	if(u < 0 || u >= x || v < 0 || v >= y)
//		return;
//
//	f[u + x * v] = fPadded[idx];
//}

__global__ void removeZeroPadding2D(float *fPadded, float *f, int paddedLen, int x, int y){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= paddedLen * paddedLen)
		return;

	int u_padding = (paddedLen - x) / 2;
	int v_padding = (paddedLen - y) / 2;
	int u = idx % paddedLen - u_padding;
	int v = idx / paddedLen - v_padding;

	if (u < 0 || u >= x || v < 0 || v >= y)
		return;

	f[u + x * v] = fPadded[idx];
	//f[v * x + u] = fPadded[idx];
}


__global__ void realToComplex(float *f, cufftComplex *fc, int len){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len)
		return;

	fc[idx].x = f[idx];
	fc[idx].y = 0.0f;
}

__global__ void complexToReal(cufftComplex *fc, float *f, int len){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len)
		return;

	f[idx] = fc[idx].x / len;
}