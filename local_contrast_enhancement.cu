#include <iostream>
#include <stdio.h>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#define NUM_STREAM 4
#define WINSIZE 32

using namespace std;
using namespace cv;

Mat read_BMP_opencv(char* filename, int& w, int& h);

__global__ void meanStddev2D(uchar *g_idata, float *g_mean, float *g_stddev, int width, int height);
__global__ void localContrastEnhance(uchar *g_idata, uchar *g_odata, float *g_mean, float *g_stddev, int width, int height);


Mat read_BMP_opencv(char* filename, int& w, int& h)
{
	Mat input_img = imread(filename, 0);
	if (input_img.empty())
		throw "Argument Exception";

	// extract image height and width from header
	int width = input_img.cols;
	int height = input_img.rows;

	//cout << endl;
	//cout << "  Name: " << filename << endl;
	//cout << " Width: " << width << endl;
	//cout << "Height: " << height << endl;

	w = width;
	h = height;

	return input_img;
}


int main()
{
	cudaEvent_t start, stop;
	float  elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));


	int f_width, f_height;
	Mat img = read_BMP_opencv("input_images/enhancetest_512.bmp", f_width, f_height);
	
	uchar* data;
	uchar* d_data;
	uchar* d_enhanced;
	float *d_mean;
	float *d_stddev;
	float *h_mean;
	float *h_stddev;
	uchar *h_enhanced;

	cudaMalloc((void**)&d_data, sizeof(uchar)*f_width*f_height);
	cudaMalloc((void**)&d_enhanced, sizeof(uchar)*f_width*f_height);
	cudaMalloc((void**)&d_mean, sizeof(float)*f_width*f_height/(WINSIZE*WINSIZE));
	cudaMalloc((void**)&d_stddev, sizeof(float)*f_width*f_height/(WINSIZE*WINSIZE));

	cudaMallocHost((void**)&data, sizeof(uchar)*f_width*f_height);
	cudaMallocHost((void**)&h_enhanced, sizeof(uchar)*f_width*f_height);
	cudaMallocHost((void**)&h_mean, sizeof(float)*f_width*f_height/(WINSIZE*WINSIZE));
	cudaMallocHost((void**)&h_stddev, sizeof(float)*f_width*f_height/(WINSIZE*WINSIZE));

	data = img.data;

	dim3 threadsPerBlock(WINSIZE, WINSIZE, 1);
	dim3 numBlocks(int(f_width / threadsPerBlock.x), int(f_height / threadsPerBlock.y), 1);
	
	cudaMemcpy(d_data, data, sizeof(uchar)*f_width*f_height, cudaMemcpyHostToDevice);

	meanStddev2D << <numBlocks, threadsPerBlock >> > (d_data, d_mean, d_stddev, f_width, f_height);
	localContrastEnhance << <numBlocks, threadsPerBlock >> > (d_data, d_enhanced, d_mean, d_stddev, f_width, f_height);

	cudaMemcpy(h_mean, d_mean, sizeof(float)*f_width*f_height / (WINSIZE*WINSIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_stddev, d_stddev, sizeof(float)*f_width*f_height / (WINSIZE*WINSIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_enhanced, d_enhanced, sizeof(uchar)*f_width*f_height, cudaMemcpyDeviceToHost);

	Mat result = Mat(f_height, f_width, CV_8UC1);
	result.data = h_enhanced;

	imwrite("output_images/enhanced_cuda.bmp", result);

	img.convertTo(img, CV_32FC1);

	Mat dstImage(img.size(), CV_8UC1, Scalar(0));
	int num = 0;
	for (int y = 0; y < img.rows; y += WINSIZE)
	{
		for (int x = 0; x < img.cols; x += WINSIZE)
		{
			Mat ROI;
			int x0 = x, y0 = y;
			if (img.cols < x + WINSIZE)
			{
				x0 = img.cols - WINSIZE;
			}
			if (img.rows < y + WINSIZE)
			{
				y0 = img.rows - WINSIZE;
			}
			img(Range(y0, y0 + WINSIZE), Range(x0, x0 + WINSIZE)).copyTo(ROI);

			Scalar avg, stddev;
			meanStdDev(ROI, avg, stddev);


			double D = 128;
			ROI = ROI - avg[0];
			ROI = ROI / (stddev[0] * 1);
			ROI = ROI * D;
			ROI = ROI + avg[0];

			Mat rst;
			ROI.convertTo(rst, CV_8UC1);

			char save_name[260];
			//sprintf_s(save_name, "save/%d.bmp", num++);
			//sprintf_s(save_name, "save/(%d, %d).bmp", x, y);
			//imwrite(save_name, rst);

			rst.copyTo(dstImage(Range(y0, y0 + WINSIZE), Range(x0, x0 + WINSIZE)));
		}
	}
	imwrite("output_images/enhanced_cv_cpu.bmp", dstImage);


	return 0;
}

__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}


__global__ void meanStddev2D(uchar *g_idata, float *g_mean, float *g_stddev, int width, int height)
{
	__shared__ int sdata[WINSIZE*WINSIZE];
	__shared__ int sdata2[WINSIZE*WINSIZE];
	// each thread loads one element from global to shared mem

	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index_in = yIndex*width + xIndex;


	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		

	sdata[tid] = g_idata[index_in];
	sdata2[tid] = g_idata[index_in]*g_idata[index_in];
	__syncthreads();

	// do reduction in shared mem
	if (tid < 512) { sdata[tid] += sdata[tid + 512];
					 sdata2[tid] += sdata2[tid + 512]; 
	}
	__syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256];
					 sdata2[tid] += sdata2[tid + 256]; 
	}
	__syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128];
					 sdata2[tid] += sdata2[tid + 128]; 
	}
	__syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64];
					sdata2[tid] += sdata2[tid + 64]; 
	}
	__syncthreads();
	if (tid < 32) { warpReduce(sdata, tid);
					warpReduce(sdata2, tid);
	}
	// write result for this block to global mem
	if (tid == 0) {
		g_mean[blockIdx.y * gridDim.x + blockIdx.x] = (float)sdata[0] / (float)(WINSIZE*WINSIZE);
		g_stddev[blockIdx.y * gridDim.x + blockIdx.x] = sqrtf(((float)sdata2[0] / (float)(WINSIZE*WINSIZE)) - ((float)sdata[0] / (float)(WINSIZE*WINSIZE))*((float)sdata[0] / (float)(WINSIZE*WINSIZE)));
	}
}


__global__ void localContrastEnhance(uchar *g_idata, uchar *g_odata, float *g_mean, float *g_stddev, int width, int height)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int meanStddevIndex = blockIdx.y * gridDim.x + blockIdx.x;

	float val = (float)g_mean[meanStddevIndex] + (128.0f / (float)g_stddev[meanStddevIndex])*((float)g_idata[yIndex*width + xIndex] - g_mean[meanStddevIndex]);
	__syncthreads();
	//if (val < 0.0f) val = fmaxf(0, val);
	val = fmaxf(0.f, val);
	__syncthreads();
	//if (val > 255.0f) val = fminf(val, 255);
	val = fminf(val, 255.f);
	__syncthreads();

	g_odata[yIndex*height + xIndex] = (uchar)val;
}