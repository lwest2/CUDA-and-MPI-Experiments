/*
 * This file contains C routines for the MPI Casestudy.
 *
 * "pgmread" reads in a PGM picture and can be called as follows:
 *
 *    float buf[M][N];
 *    pgmread("edge.pgm", buf, M, N);
 *
 * "pgmwrite" writes an array as PGM picture and can be called as follows:
 *
 *    float buf[M][N];
 *    pgmwrite("picture.pgm", buf, M, N);
 *
 * "pgmsize" returns the size of a PGM picture and can be called as follows:
 *
 *    int nx, ny;
 *    pgmsize("edge.pgm", &nx, &ny);
 *
 *  To access these routines, add the following to your program:
 *
 *    #include "pgmio.h"
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAXLINE 128

/*
 *  Routine to get the size of a PGM data file
 *
 *  Note that this assumes a single line comment and no other white space.
 */

void pgmsize(char *filename, int *nx, int *ny)
{ 
  FILE *fp;

  char *cret;
  int iret;

  char dummy[MAXLINE];
  int n = MAXLINE;

  fopen_s(&fp,filename,"r");

  if (fp == NULL)
  {
    fprintf(stderr, "pgmsize: cannot open <%s>\n", filename);
    exit(-1);
  }

  cret = fgets(dummy, n, fp);
  cret = fgets(dummy, n, fp);

  iret = fscanf_s(fp,"%d %d", nx, ny);
      
  fclose(fp);
}


/*
 *  Routine to read a PGM data file into a 2D floating point array
 *  x[nx][ny]. Because of the way C handles (or fails to handle!)
 *  multi-dimensional arrays we have to cast the pointer to void.
 *
 *  Note that this assumes a single line comment and no other white space.
 */

void pgmread(char *filename, void *vx, int nx, int ny)
{ 
  FILE *fp;

  int nxt, nyt, i, j, t;
  char dummy[MAXLINE];
  int n = MAXLINE;

  char *cret;
  int iret;

  float *x = (float *) vx;

  fopen_s(&fp, filename, "r");

  if (fp == NULL)
  {
    fprintf(stderr, "pgmread: cannot open <%s>\n", filename);
    exit(-1);
  }
  
  cret = fgets(dummy, n, fp);
  cret = fgets(dummy, n, fp);

  iret = fscanf_s(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "pgmread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  iret = fscanf_s(fp,"%d",&i);

  /*
   *  Must cope with the fact that the storage order of the data file
   *  is not the same as the storage of a C array, hence the pointer
   *  arithmetic to access x[i][j].
   */

  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
      iret = fscanf_s(fp,"%d", &t);
      x[(ny-j-1)+ny*i] = t;
    }
  }

  fclose(fp);
}



/*
*  Routine to read a DAT data file into a 2D floating point array
*  x[nx][ny]. Because of the way C handles (or fails to handle!)
*  multi-dimensional arrays we have to cast the pointer to void.
*
*  Note that this assumes a single line comment and no other white space.
*/void datread(char *filename, void *vx, int nx, int ny)
{
	FILE *fp;

	int nxt, nyt, i, j, t;

	float *x = (float *)vx;

	if (NULL == (fp = fopen(filename, "r")))
	{
		fprintf(stderr, "datread: cannot open <%s>\n", filename);
		exit(-1);
	}

	fscanf(fp, "%d %d", &nxt, &nyt);

	if (nx != nxt || ny != nyt)
	{
		fprintf(stderr,
			"datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
			nxt, nyt, nx, ny);
		exit(-1);
	}

	for (j = 0; j<ny; j++)
	{
		for (i = 0; i<nx; i++)
		{
			fscanf(fp, "%d", &t);
			x[(ny - j - 1)*nx + i] = t;
		}
	}

	fclose(fp);
}

/*
*  Routine to write a PGM image file from a 2D floating point array
*  x[nx][ny]. Because of the way C handles (or fails to handle!)
*  multi-dimensional arrays we have to cast the pointer to void.
*/
void pgmwrite(char *filename, void *vx, int nx, int ny)
{
	FILE *fp;

	int i, j, k, grey;

	float xmin, xmax, tmp;
	float thresh = 255.0;

	float *x = (float *)vx;

	if (NULL == (fp = fopen(filename, "w")))
	{
		fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
		exit(-1);
	}

	/*
	*  Find the max and min absolute values of the array
	*/

	xmin = fabs(x[0]);
	xmax = fabs(x[0]);

	for (i = 0; i < nx*ny; i++)
	{
		if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
		if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
	}

	fprintf(fp, "P2\n");
	fprintf(fp, "# Written by pgmwrite\n");
	fprintf(fp, "%d %d\n", nx, ny);
	fprintf(fp, "%d\n", (int)thresh);

	k = 0;

	for (j = ny - 1; j >= 0; j--)
	{
		for (i = 0; i < nx; i++)
		{
			/*
			*  Access the value of x[i][j]
			*/

			tmp = x[j*nx + i];

			/*
			*  Scale the value appropriately so it lies between 0 and thresh
			*/

			if (xmin < 0 || xmax > thresh)
			{
				tmp = (int)((thresh*((fabs(tmp - xmin)) / (xmax - xmin))) + 0.5);
			}
			else
			{
				tmp = (int)(fabs(tmp) + 0.5);
			}

			/*
			*  Increase the contrast by boosting the lower values
			*/

			grey = (int)(thresh * sqrt(tmp / thresh));

			fprintf(fp, "%3d ", grey);

			if (0 == (k + 1) % 16) fprintf(fp, "\n");

			k++;
		}
	}

	if (0 != k % 16) fprintf(fp, "\n");
	fclose(fp);
}
