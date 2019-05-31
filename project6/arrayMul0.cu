// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

#ifndef SIZE
#define SIZE			1*1024*1024	// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100		// to make the timing more accurate
#endif

#ifndef NUMTRIES
#define NUMTRIES	10
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

float
Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int
Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}

void
TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}

__global__  void ArrayMul( float *A, float *B, float *C )
{
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	prods[tnum] = A[gid] * B[gid];

	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			prods[tnum] += prods[tnum + offset];
		}
	}

	__syncthreads();
	if (tnum == 0)
		C[wgNum] = prods[0];
}


// main program:

int
main( int argc, char* argv[ ] )
{
	FILE *f;
	f = fopen("results.txt","a");
	int dev = findCudaDevice(argc, (const char **)argv);
	TimeOfDaySeed( );		// seed the random number generator

	// allocate host memory:

	float * hXCS = new float [ NUMTRIALS ];
  float * hYCS = new float [ NUMTRIALS ];
  float * hRS = new float [ NUMTRIALS ];
  float * hT = new float [ NUMTRIALS ];

	for( int i = 0; i < SIZE; i++ )
	{
		hXCS[n] = Ranf( XCMIN, XCMAX );
		hYCS[n] = Ranf( YCMIN, YCMAX );
		hRS[n] = Ranf(  RMIN,  RMAX );
	}

	// allocate device memory:

	float *dXCS, *dYCS, *dRS, *dT;

  dim3 dimsXCS( NUMTRIALS, 1, 1 );
  dim3 dimsYCS( NUMTRIALS, 1, 1 );
  dim3 dimsRS( NUMTRIALS, 1, 1 );
  dim3 dimsT( NUMTRIALS, 1, 1 );

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dXCS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dYCS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dRS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dT), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );


	// copy host memory to the device:

	status = cudaMemcpy( dXCS, hXCS, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
  checkCudaErrors( status );
  status = cudaMemcpy( dYCS, hYCS, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
  checkCudaErrors( status );
  status = cudaMemcpy( dRS, hRS, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
  checkCudaErrors( status );

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
  dim3 grid( NUMTRIALS / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:

	for( int t = 0; t < NUMTRIALS; t++)
	{
	        LaserMonteCarlo<<< grid, threads >>>( dXCS, dYCS, dRS, dT );
	}

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf( stderr, "Array Size = %10d, MegaMultReductions/Second = %10.2lf\n", SIZE, megaMultsPerSecond );

	// copy result from the device to the host:

	status = cudaMemcpy( hC, dC, (SIZE/BLOCKSIZE)*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum :

					// double sum = 0.;
					// for(int i = 0; i < SIZE/BLOCKSIZE; i++ )
					// {
					// 	//fprintf(stderr, "hC[%6d] = %10.2f\n", i, hC[i]);
					// 	sum += (double)hC[i];
					// }
					// fprintf( stderr, "\nsum = %10.2lf\n", sum );

	// clean up memory:
	delete [ ] hXCS;
	delete [ ] hYCS;
	delete [ ] hRS;
	delete [ ] hT;

	status = cudaFree( dXCS );
		checkCudaErrors( status );
	status = cudaFree( dYCS );
		checkCudaErrors( status );
	status = cudaFree( dRS );
		checkCudaErrors( status );
	status = cudaFree( dT );
		checkCudaErrors( status );

	return 0;
}
