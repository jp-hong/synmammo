

#include "imageCalculator.cuh"


__global__ void projectionAndMIP(float *projection, float *mip, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nu * nv)
		return;

	int jp = idx % nu;
	int kp = idx / nu;

	int i, ix, iy, iz;

	float xSource, ySource, zSource, xDetCenter, yDetCenter, zDetCenter,
		eux, euy, euz, evx, evy, evz, u, v, xbin, ybin, zbin, x, y, z,
		xdiff, ydiff, zdiff, xad, yad, zad, xoy, xoz, yox, yoz, zox, zoy, travVoxlen, total, maxval, val0;

	__shared__ float s_frame_vectors[12];

	if(threadIdx.x == 0){
		for(i = 0; i < 12; i++)
			s_frame_vectors[i] = frame_vectors[i];
	}

	__syncthreads();

	xSource = s_frame_vectors[0];
	ySource = s_frame_vectors[1];
	zSource = s_frame_vectors[2];

	xDetCenter = s_frame_vectors[3];
	yDetCenter = s_frame_vectors[4];
	zDetCenter = s_frame_vectors[5];

	eux = s_frame_vectors[6];
	euy = s_frame_vectors[7];
	euz = s_frame_vectors[8];

	evx = s_frame_vectors[9];
	evy = s_frame_vectors[10];
	evz = s_frame_vectors[11];

	u = u0 + (jp + 0.5f) * du;
	v = v0 + (kp + 0.5f) * dv;
	
	xbin = xDetCenter + eux * u + evx * v;
	ybin = yDetCenter + euy * u + evy * v;
	zbin = zDetCenter + euz * u + evz * v;

	xdiff = xbin - xSource;
	ydiff = ybin - ySource;
	zdiff = zbin - zSource;

	xad = fabsf(xdiff) / dx;
	yad = fabsf(ydiff) / dy;
	zad = fabsf(zdiff) / dz;

	total = 0.0f;
	maxval = -10000.0f;
	
	if(xad > yad && xad > zad){
		yox = ydiff / xdiff;
		zox = zdiff / xdiff;
		travVoxlen = dx * sqrtf(1 + yox * yox + zox * zox);

		for(ix = 0; ix < nx; ix++){
			x = x0 + dx * (ix + 0.5);
			y = ySource + yox * (x - xSource);
			z = zSource + zox * (x - xSource);
			iy = (int)floorf((y - y0) / dy - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(iy >= 0 && iy < ny - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
				//maxval = MAX(val0, maxval);
				maxval = MAX(travVoxlen * val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}else if(yad > zad){
		xoy = xdiff / ydiff;
		zoy = zdiff / ydiff;
		travVoxlen = dy * sqrtf(1 + xoy * xoy + zoy * zoy);

		for(iy = 0; iy < ny; iy++){
			y = y0 + dy * (iy + 0.5f);
			x = xSource + xoy * (y - ySource);
			z = zSource + zoy * (y - ySource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
				//maxval = MAX(val0, maxval);
				maxval = MAX(travVoxlen * val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}else{
		xoz = xdiff / zdiff;
		yoz = ydiff / zdiff;
		travVoxlen = dz * sqrtf(1 + xoz * xoz + yoz * yoz);

		for(iz = 0; iz < nz; iz++){
			z = z0 + dz * (iz + 0.5f);
			x = xSource + xoz * (z - zSource);
			y = ySource + yoz * (z - zSource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iy = (int)floorf((y - y0) / dy - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
				//maxval = MAX(val0, maxval);
				maxval = MAX(travVoxlen * val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}

	projection[jp + nu * (nv - 1 - kp)] = total;
	mip[jp + nu * (nv - 1 - kp)] = maxval;
}

__global__ void sinoproj(float *sinomat, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nu * nv)
		return;

	int jp = idx % nu;
	int kp = idx / nu;

	int i, ix, iy, iz;

	float xSource, ySource, zSource, xDetCenter, yDetCenter, zDetCenter,
		eux, euy, euz, evx, evy, evz, u, v, xbin, ybin, zbin, x, y, z,
		xdiff, ydiff, zdiff, xad, yad, zad, xoy, xoz, yox, yoz, zox, zoy, travVoxlen, total, val0;

	__shared__ float s_frame_vectors[12];

	if(threadIdx.x == 0){
		for(i = 0; i < 12; i++)
			s_frame_vectors[i] = frame_vectors[i];
	}

	__syncthreads();

	xSource = s_frame_vectors[0];
	ySource = s_frame_vectors[1];
	zSource = s_frame_vectors[2];

	xDetCenter = s_frame_vectors[3];
	yDetCenter = s_frame_vectors[4];
	zDetCenter = s_frame_vectors[5];

	eux = s_frame_vectors[6];
	euy = s_frame_vectors[7];
	euz = s_frame_vectors[8];

	evx = s_frame_vectors[9];
	evy = s_frame_vectors[10];
	evz = s_frame_vectors[11];

	u = u0 + (jp + 0.5f) * du;
	v = v0 + (kp + 0.5f) * dv;
	
	xbin = xDetCenter + eux * u + evx * v;
	ybin = yDetCenter + euy * u + evy * v;
	zbin = zDetCenter + euz * u + evz * v;

	xdiff = xbin - xSource;
	ydiff = ybin - ySource;
	zdiff = zbin - zSource;

	xad = fabsf(xdiff) / dx;
	yad = fabsf(ydiff) / dy;
	zad = fabsf(zdiff) / dz;

	total = 0.0f;
	
	if(xad > yad && xad > zad){
		yox = ydiff / xdiff;
		zox = zdiff / xdiff;
		travVoxlen = dx * sqrtf(1 + yox * yox + zox * zox);

		for(ix = 0; ix < nx; ix++){
			x = x0 + dx * (ix + 0.5);
			y = ySource + yox * (x - xSource);
			z = zSource + zox * (x - xSource);
			iy = (int)floorf((y - y0) / dy - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(iy >= 0 && iy < ny - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
			}
		}
	}else if(yad > zad){
		xoy = xdiff / ydiff;
		zoy = zdiff / ydiff;
		travVoxlen = dy * sqrtf(1 + xoy * xoy + zoy * zoy);

		for(iy = 0; iy < ny; iy++){
			y = y0 + dy * (iy + 0.5f);
			x = xSource + xoy * (y - ySource);
			z = zSource + zoy * (y - ySource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
			}
		}
	}else{
		xoz = xdiff / zdiff;
		yoz = ydiff / zdiff;
		travVoxlen = dz * sqrtf(1 + xoz * xoz + yoz * yoz);

		for(iz = 0; iz < nz; iz++){
			z = z0 + dz * (iz + 0.5f);
			x = xSource + xoz * (z - zSource);
			y = ySource + yoz * (z - zSource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iy = (int)floorf((y - y0) / dy - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				total += travVoxlen * val0;
			}
		}
	}

	sinomat[jp + nu * (nv - 1 - kp)] = total;
}

__global__ void sinoproj_max(float *sinomat_max, float *frame_vectors, int nu, int nv, float du, float dv, float u0, float v0,
	const short *smat, float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nu * nv)
		return;

	int jp = idx % nu;
	int kp = idx / nu;

	int i, ix, iy, iz;

	float xSource, ySource, zSource, xDetCenter, yDetCenter, zDetCenter,
		eux, euy, euz, evx, evy, evz, u, v, xbin, ybin, zbin, x, y, z,
		xdiff, ydiff, zdiff, xad, yad, zad, xoy, xoz, yox, yoz, zox, zoy, val0, maxval;

	__shared__ float s_frame_vectors[12];

	if(threadIdx.x == 0){
		for(i = 0; i < 12; i++)
			s_frame_vectors[i] = frame_vectors[i];
	}

	__syncthreads();

	xSource = s_frame_vectors[0];
	ySource = s_frame_vectors[1];
	zSource = s_frame_vectors[2];

	xDetCenter = s_frame_vectors[3];
	yDetCenter = s_frame_vectors[4];
	zDetCenter = s_frame_vectors[5];

	eux = s_frame_vectors[6];
	euy = s_frame_vectors[7];
	euz = s_frame_vectors[8];

	evx = s_frame_vectors[9];
	evy = s_frame_vectors[10];
	evz = s_frame_vectors[11];

	u = u0 + (jp + 0.5f) * du;
	v = v0 + (kp + 0.5f) * dv;
	
	xbin = xDetCenter + eux * u + evx * v;
	ybin = yDetCenter + euy * u + evy * v;
	zbin = zDetCenter + euz * u + evz * v;

	xdiff = xbin - xSource;
	ydiff = ybin - ySource;
	zdiff = zbin - zSource;

	xad = fabsf(xdiff) / dx;
	yad = fabsf(ydiff) / dy;
	zad = fabsf(zdiff) / dz;

	maxval = -10000.0f;
	
	if(xad > yad && xad > zad){
		yox = ydiff / xdiff;
		zox = zdiff / xdiff;

		for(ix = 0; ix < nx; ix++){
			x = x0 + dx * (ix + 0.5);
			y = ySource + yox * (x - xSource);
			z = zSource + zox * (x - xSource);
			iy = (int)floorf((y - y0) / dy - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(iy >= 0 && iy < ny - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				maxval = MAX(val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}else if(yad > zad){
		xoy = xdiff / ydiff;
		zoy = zdiff / ydiff;

		for(iy = 0; iy < ny; iy++){
			y = y0 + dy * (iy + 0.5f);
			x = xSource + xoy * (y - ySource);
			z = zSource + zoy * (y - ySource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iz = (int)floorf((z - z0) / dz - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iz >= 0 && iz < nz - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				maxval = MAX(val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}else{
		xoz = xdiff / zdiff;
		yoz = ydiff / zdiff;

		for(iz = 0; iz < nz; iz++){
			z = z0 + dz * (iz + 0.5f);
			x = xSource + xoz * (z - zSource);
			y = ySource + yoz * (z - zSource);
			ix = (int)floorf((x - x0) / dx - 0.5f);
			iy = (int)floorf((y - y0) / dy - 0.5f);

			if(ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1){
				val0 = smat[nz * ny * ix + ny * (nz - 1 - iz) + iy];
				maxval = MAX(val0, maxval);
			}
		}

		if(maxval == -10000.0f)
			maxval = 0;
	}

	sinomat_max[jp + nu * (nv - 1 - kp)] = maxval;
}

__global__ void combineImages(const float *sino, const float *sinoMax, float *sinoResult, int len, float scaleFactor){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len)
		return;

	sinoResult[idx] = sino[idx] + sinoMax[idx] * scaleFactor;
}

__global__ void offsetImage(float *f, int len, float offset){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len)
		return;

	f[idx] += offset;
}

float mean(float *f, int len){
	float *d_sum, h_sum;

	cudaMalloc((void**)&d_sum, sizeof(float));
	cudaMemset(d_sum, 0, sizeof(float));

	dim3 BLOCK(256);
	dim3 GRID(0);

	getGridDim(&GRID, BLOCK, len);
	sumKernel <<<GRID, BLOCK>>> (f, len, d_sum);
	cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_sum);

	return h_sum / (float)len;
}

__global__ void sumKernel(float *f, int len, float *globalSum){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len) return;

	__shared__ float s_sum;

	if(threadIdx.x == 0)
		s_sum = 0.0f;

	__syncthreads();
	atomicAdd(&s_sum, f[idx]);
	__syncthreads();

	if(threadIdx.x == 0)
		atomicAdd(globalSum, s_sum);
}

float standardDeviation(float *f, int len){
	float *d_sum, h_sum;

	cudaMalloc((void**)&d_sum, sizeof(float));
	cudaMemset(d_sum, 0, sizeof(float));

	float avg = mean(f, len);

	dim3 BLOCK(256);
	dim3 GRID(0);
	
	getGridDim(&GRID, BLOCK, len);
	standardDeviationKernel <<<GRID, BLOCK>>> (f, len, avg, d_sum);
	cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_sum);

	return sqrtf(h_sum);
}

__global__ void standardDeviationKernel(float *f, int len, float mean, float *globalSum){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len) return;

	__shared__ float s_sum;

	if(threadIdx.x == 0)
		s_sum = 0.0f;

	__syncthreads();
	atomicAdd(&s_sum, (f[idx] - mean) * (f[idx] - mean));
	__syncthreads();

	if(threadIdx.x == 0)
		atomicAdd(globalSum, s_sum);
}

__global__ void forceNonNegativity(float *f, int len){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len) return;

	if(f[idx] < 0) f[idx] = 0;
}