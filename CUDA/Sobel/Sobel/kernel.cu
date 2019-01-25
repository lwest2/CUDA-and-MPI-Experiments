#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

// image dimensions WIDTH & HEIGHT
#define WIDTH 256
#define HEIGHT 256

// Block width WIDTH & HEIGHT
#define BLOCK_W 16
#define BLOCK_H 16

// buffer to read image into
float image[HEIGHT][WIDTH];

// buffer for resulting image
float final[HEIGHT][WIDTH];

// prototype declarations
void load_image();
void call_kernel();
void save_image();

// Times used to calculate performance
double start_time_blur, end_time_blur;
double start_time_sobel, end_time_sobel;
double start_time_total, end_time_total;

// pgm file reader header file
extern "C" {
#include "pgmio.h"
}

// time header file
#include "win-gettimeofday.h"

// Kernel for image bluring before sobel filter operation
__global__ void imageBlur(float *input, float *output, int width, int height) {
	
	// index of thread
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	// width of image
	int numcols = WIDTH;

	// float for pixel
	float blur;

	// if within the bounds of the image
	if (row <= height && col <= width && row > 0 && col > 0)
	{
		// weights
		int		x1,
			x3, x4, x5,
				x7;

		// blur
		// 0.0 0.2 0.0
		// 0.2 0.2 0.2
		// 0.0 0.2 0.0

		// get index of mask & pixel to edit
		x1 = input[(row + 1) * numcols + col];			// up
		x3 = input[row * numcols + (col - 1)];			// left
		x4 = input[row * numcols + col];				// center
		x5 = input[row * numcols + (col + 1)];			// right
		x7 = input[(row + -1) * numcols + col];			// down


		// calculate blur
		blur = (x1 * 0.2) + (x3 * 0.2) + (x4 * 0.2) + (x5 * 0.2) + (x7 * 0.2);

		// output pixel data
		output[row * numcols + col] = blur;
	}
}

// sobel filter
__global__ void sobelFilter(float *input, float *output, int width, int height) {
	
	// index of thread
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	/// width of image
	int numcols = WIDTH;

	float gradient_h;	// horizontal gradient
	float gradient_v;	// vertical gradient
	float gradient;		// combined gradient
	float thresh = 30;	// threshold value

	// if within the bounds of the image
	if (row <= height && col <= width && row > 0 && col > 0)
	{
		// weights
		int x0, x1, x2, 
		    x3,	    x5, 
			x6, x7, x8;

		// horizontal
		// -1  0  1
		// -2  0  2
		// -1  0  1

		// vertical 
		// -1 -2 -1
		//  0  0  0
		//  1  2  1

		// get index of mask & pixel to edit
		x0 = input[(row - 1) * numcols + (col - 1)];	// leftup
		x1 = input[(row + 1) * numcols + col];			// up
		x2 = input[(row - 1) * numcols + (col + 1)];	// rightup
		x3 = input[row * numcols + (col - 1)];			// left
		x5 = input[row * numcols + (col + 1)];			// right
		x6 = input[(row + 1) * numcols + (col - 1)];	// leftdown
		x7 = input[(row + -1) * numcols + col];			// down
		x8 = input[(row + 1) * numcols + (col + 1)];	// rightdown


		// calculate gradients for horizontal and vertical axis using sobel mask
		gradient_h = (x0 * -1) + (x2 * 1) + (x3 * -2) + (x5 * 2) + (x6 * -1) + (x8 * 1);
		gradient_v = (x0 * -1) + (x1 * -2) + (x3 * -1) + (x6 * 1) + (x7 * 2) + (x8 * 1);

		// using pythagoras theorem calculate the total gradient
		gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

		if (gradient >= thresh)
		{
			gradient = 255;
		}
		else {
			gradient = 0;
		}
		// output pixel data
		output[row * numcols + col] = gradient;
	}
}

void load_image() {
	// read pgm file
	pgmread("image512x512.pgm", (void *)image, WIDTH, HEIGHT);
	// write read data for test
	pgmwrite("image_test512x512.pgm", (void *)image, WIDTH, HEIGHT);
}

void save_image() {
	// save pgm file
	pgmwrite("final512x512.pgm", (void *)final, WIDTH, HEIGHT);
}

void call_kernel() {
	int x, y;
	float *d_input, *d_output;

	printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

	// memory size to allocate
	size_t memSize = WIDTH * HEIGHT * sizeof(float);

	// allocate memory
	cudaMalloc(&d_input, memSize);
	cudaMalloc(&d_output, memSize);

	// set value of pixels to 0
	for (y = 0; y < HEIGHT; y++) {
		for (x = 0; x < WIDTH; x++) {
			final[x][y] = 0.0;
		}
	}

	printf("Blocks per grid (width): %d\n", (WIDTH / BLOCK_W));
	printf("Blocks per grid (height): %d\n", (HEIGHT / BLOCK_H));

	// start time for blur
	start_time_blur = get_current_time();

	// copy image data into host input
	cudaMemcpy(d_input, image, memSize, cudaMemcpyHostToDevice);

	// call kernel blur
	dim3 threads(BLOCK_W, BLOCK_H); // threads per block
	dim3 blocks(WIDTH / BLOCK_W, HEIGHT / BLOCK_H); // blocks per grid 
	imageBlur << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);

	// sync threads
	cudaThreadSynchronize();

	// copy output back into input
	cudaMemcpy(d_input, d_output, memSize, cudaMemcpyDeviceToHost);

	// end time of blur filter
	end_time_blur = get_current_time();

	// start time for sobel filter
	start_time_sobel = get_current_time();

	// call sobel kernel with same threads and blocks amount
	sobelFilter << <blocks, threads >> > (d_input, d_output, WIDTH, HEIGHT);

	// sync threads
	cudaThreadSynchronize();

	// get output of kernel
	cudaMemcpy(final, d_output, memSize, cudaMemcpyDeviceToHost);

	// end time for sobel filter
	end_time_sobel = get_current_time();

	// if an error, display
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "Main Loop", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// free memory
	cudaFree(d_input);
	cudaFree(d_output);
}

// main method to call methods
int main(int argc, char *argv[])
{
	// total time (parallel + serial)
	start_time_total = get_current_time();

	// firstly load image into buffer
	load_image();

	// invoke kernels
	call_kernel();

	// save output
	save_image();
	
	// end total time (parallel + serial)
	end_time_total = get_current_time();

	// print times
	double total = (end_time_sobel - start_time_sobel) + (end_time_blur - start_time_blur);

	printf("GPU blur (Including Data Transfer): %fs\n", end_time_blur - start_time_blur);
	printf("GPU sobel (Including Data Transfer): %fs\n", end_time_sobel - start_time_sobel);

	printf("Total P Time: %fs\n", total);
	printf("Total S Time: %fs\n", end_time_total - total);
	printf("Total S + P Time: %fs\n", end_time_total);

	// reset device
	cudaDeviceReset();
	
	// exit with code 0
	return 0;
}