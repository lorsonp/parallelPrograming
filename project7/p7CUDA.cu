#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE 32 // number of threads per block
#endif


__global__ void AutoCorrelate( float *Array, float *Sums )
{
	// int Size = get_global_size( 0 );
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int shift = gid;
	int Size = 32769;
	float sum = 0.;
	for( int i = 0; i < Size; i++ )
	{
		sum += Array[i] * Array[i + shift];
	}
	Sums[shift] = sum;
}



int
main( int argc, char *argv[ ] )
{

  int dev = findCudaDevice(argc, (const char **)argv);


  FILE *fp = fopen( "signal.txt", "r" );
  if( fp == NULL )
  {
    fprintf( stderr, "Cannot open file 'signal.txt'\n" );
    exit( 1 );
  }
  int Size;
  fscanf( fp, "%d", &Size );
  float *Array = new float[ 2*Size ];
  float *Sums  = new float[ 1*Size ];
  for( int i = 0; i < Size; i++ )
  {
    fscanf( fp, "%f", &Array[i] );
    Array[i+Size] = Array[i];		// duplicate the array
  }
  fclose( fp );

  float *hArray = new float[ 2*Size ];
  float *hSums  = new float[ 1*Size ];
	// float *hSize  = new float[ 1 ];


  for( int n = 0; n < Size; n++ )
  {
    hArray[n] = Array[n];
    hArray[n+Size] = Array[n];
  }
	// hSize = Size

    // allocate device memory:


  float *dArray, *dSums; //, *dSize;

  dim3 dimsArray( 2*Size, 1, 1 );
  dim3 dimsSums ( 1*Size, 1, 1 );
	// dim3 dimsSize ( 1, 1, 1 );


  cudaError_t status;
  status = cudaMalloc( reinterpret_cast<void **>(&dArray), 2*Size*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dSums), 1*Size*sizeof(float) );
  checkCudaErrors( status );
	// status = cudaMalloc( reinterpret_cast<void **>(&dSize), 1*sizeof(float) );
	// checkCudaErrors( status );


    // copy host memory to the device:

  status = cudaMemcpy( dArray, hArray, 2*Size*sizeof(float), cudaMemcpyHostToDevice );
  checkCudaErrors( status );


  // setup the execution parameters:

  dim3 threads(BLOCKSIZE, 1, 1 );
  dim3 grid( Size / threads.x, 1, 1 );


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


            AutoCorrelate<<< grid, threads >>>( dArray, dSums);


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
  double megaCalcssPerSecond =   Size / secondsTotal / 1000000.;


  // copy result from the device to the host:

  status = cudaMemcpy( hSums, dSums, Size*sizeof(float), cudaMemcpyDeviceToHost );
    checkCudaErrors( status );


    FILE *f;
    // f = fopen("SumsVShift.txt","a");
		f = fopen("p7CUDA.txt","a");
    if (BLOCKSIZE == 1)
    {
      // fprintf(f,"\n");
      for (size_t i = 1; i < 512; i++)
      {
      fprintf(f,"%f  %d \n", hSums[i], i);
      }
    }

    // FILE *f1;
    fprintf(f,"%d  %f \n", BLOCKSIZE, megaCalcssPerSecond);


    delete [ ] hArray;
    delete [ ] hSums;


    status = cudaFree( dArray );
      checkCudaErrors( status );
    status = cudaFree( dSums );
      checkCudaErrors( status );
return 0;
}
