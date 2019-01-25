#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

extern "C" {
#include "pgmio.h"
}

// image size
#define M 250
#define N 360

// number of processors
#define P 4

// define chunk that image will be split into
#define MP M/P
#define NP N

#define THRESH 100

double start_time_sobel, end_time_sobel;
double start_time_total, end_time_total;

int main(int argc, char **argv)
{

	// buffers containing halo
	float old[MP + 2][NP + 2], image[MP + 2][NP + 2], edge[MP + 2][NP + 2];

	float masterbuf[M][N];	// global buffer
	float buf[MP][NP];		// local buffer

	int i, j; // for loop integers
	char *filename; // name of file

	int rank, size;
	MPI_Status status;

	// Initiate MPI
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);	// number of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	// index of processor

	// if size is not equal to number of processors
	if (size != P)
	{
		// Finalize and exit with status -1
		MPI_Finalize();
		exit(-1);
	}

	// if processor is root
	if (rank == 0)
	{
		start_time_total = MPI_Wtime();
		// get file name
		char input[] = "edge256x256.pgm";
		filename = input;
		// read pgm file
		pgmread(filename, masterbuf, M, N);

		printf("width: %d \nheight: %d\nprocessors: %d\n", M, N, P);
	}

	start_time_sobel = MPI_Wtime();

	// Scatter image, split between chunks. For example 4 processors is equal to 4 vertical chunks.
	MPI_Scatter(masterbuf, MP*NP, MPI_FLOAT, buf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// local process
	for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
			// horizontal gradient
			// -1  0  1
			// -2  0  2
			// -1  0  1

			// vertical gradient
			// -1 -2 -1
			//  0  0  0
			//  1  2  1

			float gradient_h = ((-1.0 * buf[i - 1][j - 1]) + (1.0 * buf[i + 1][j - 1]) + (-2.0 * buf[i - 1][j]) + (2.0 * buf[i + 1][j]) + (-1.0 * buf[i - 1][j + 1]) + (1.0 * buf[i + 1][j + 1]));
			float gradient_v = ((-1.0 * buf[i - 1][j - 1]) + (-2.0 * buf[i][j - 1]) + (-1.0 * buf[i + 1][j - 1]) + (1.0 * buf[i - 1][j + 1]) + (2.0 * buf[i][j + 1]) + (1.0 * buf[i + 1][j + 1]));

			// calculate gradient magnitude
			float gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

			// if gradient is below threshold
			if (gradient < THRESH) {
				gradient = 0;	// set gradient to 0
			}
			else {
				gradient = 255;	// set gradient to 255
			}

			// apply pixel index the gradient
			image[i][j] = gradient;
		}
	}

	end_time_sobel = MPI_Wtime();


	if (rank == 0)
	{
		printf("Finished");
	}

	// apply finished image to the local buf without edge
	for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
			buf[i - 1][j - 1] = image[i][j];
		}
	}

	// Gather all local bufs back to the root global buf
	MPI_Gather(buf, MP*NP, MPI_FLOAT, masterbuf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);


	// on root
	if (rank == 0)
	{
		// attach file name
		char output[] = "image256x256.pgm";
		filename = output;
		// write file
		printf("\nOutput: <%s>\n", filename);
		pgmwrite(filename, masterbuf, M, N);

		end_time_total = MPI_Wtime();

		// print timescd 
		double total = (end_time_sobel - start_time_sobel);
		printf("Total P Time: %fs\n", total);
		printf("Total S Time: %fs\n", (end_time_total - start_time_total) - total);
		printf("Total S + P Time: %fs\n", end_time_total - start_time_total);

	}

	
	// Finalize MPI
	MPI_Finalize();

	// exit with status 0
	return 0;
}
