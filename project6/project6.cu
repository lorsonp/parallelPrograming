// Laser Monte Carlo


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
#define BLOCKSIZE 32 // number of threads per block
#endif

// #ifndef
//   #define SIZE 1*1024*1024 // array size
// #endif

#ifndef NUMTRIALS
#define NUMTRIALS 100 // to make the timing more accurate
#endif

#ifndef NUMTRIES
#define NUMTRIES	10
#endif

#ifndef TOLERANCE
#define TOLERANCE 0.00001f // tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN =	 0.0;
const float XCMAX =	 2.0;
const float YCMIN =	 0.0;
const float YCMAX =	 2.0;
const float RMIN  =	 0.5;
const float RMAX  =	 2.0;

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

__global__ void LaserMonteCarloNoIf( float *XCS, float *YCS, float *RS, float *T )
{
    __shared__ float prods[BLOCKSIZE];
	
    unsigned int numItems = blockDim.x;
    unsigned int tnum = threadIdx.x;
    unsigned int wgNum = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
 
   // solve for the intersection using the quadratic formula:
    float a = 2.;
    float b = -2.*( XCS[gid] + YCS[gid] );
    float c = XCS[gid]*XCS[gid] + YCS[gid]*YCS[gid] - RS[gid]*RS[gid];
    float d = b*b - 4.*a*c;
    //IF d is less than 0., then the circle was completely missed.
    //(Case A) Continue on to the next trial in the for-loop.

        // hits the circle:
        // get the first intersection:
        d = sqrt( d );
        float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
        float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
        float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection


            //IF tmin is less than 0., then the circle completely engulfs the laser pointer.
            //(Case B) Continue on to the next trial in the for-loop.

            // where does it intersect the circle?
            float xcir = tmin;
            float ycir = tmin;

            // get the unitized normal vector at the point of intersection:
            float nx = xcir - XCS[gid];
            float ny = ycir - YCS[gid];
            float n = sqrt( nx*nx + ny*ny );
            nx /= n;	// unit vector
            ny /= n;	// unit vector

            // get the unitized incoming vector:
            float inx = xcir - 0.;
            float iny = ycir - 0.;
            float in = sqrt( inx*inx + iny*iny );
            inx /= in;	// unit vector
            iny /= in;	// unit vector

            // get the outgoing (bounced) vector:
            float dot = inx*nx + iny*ny;
            // float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
            float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

            // find out if it hits the infinite plate:
            float t = ( 0. - ycir ) / outy;
            if (t > 0)
          		{
          			prods[tnum] = 1;
          		}
            else
              {
                prods[tnum] = 0;
              }

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
		T[wgNum] = prods[0];
}





// main program:

int
main( int argc, char *argv[ ] )
{
	FILE *f;
	f = fopen("results.txt","a");
	int dev = findCudaDevice(argc, (const char **)argv);

  TimeOfDaySeed( );		// seed the random number generator

  // allocate host memory:
  float * hXCS = new float [ NUMTRIALS ];
  float * hYCS = new float [ NUMTRIALS ];
  float * hRS = new float [ NUMTRIALS ];
  float * hT = new float [ NUMTRIALS/BLOCKSIZE ];

  for( int n = 0; n < NUMTRIALS; n++ )
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
  dim3 dimsT( NUMTRIALS/BLOCKSIZE, 1, 1 );

  // __shared__ float prods[NUMTRIALS/BLOCKSIZE];


  cudaError_t status;
  status = cudaMalloc( reinterpret_cast<void **>(&dXCS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dYCS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dRS), NUMTRIALS*sizeof(float) );
  checkCudaErrors( status );
  status = cudaMalloc( reinterpret_cast<void **>(&dT), NUMTRIALS/BLOCKSIZE*sizeof(float) );
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

        // for( int t = 0; t < NUMTRIES; t++)
        // {
            LaserMonteCarloNoIf<<< grid, threads >>>( dXCS, dYCS, dRS, dT );
        // }

	

  // record the stop event:

  status = cudaEventRecord( stop, NULL );
    checkCudaErrors( status );

  // wait for the stop event to complete:

  status = cudaEventSynchronize( stop );
    checkCudaErrors( status );
  // sum hits
  //  int hits;
  //  for (int i = 0; i < NUMTRIALS; i++)
  //  {
  //	hits += hT[i];
  //  }
  //  double P = (double)hits/(float)NUMTRIALS;


  float msecTotal = 0.0f;
  status = cudaEventElapsedTime( &msecTotal, start, stop );
    checkCudaErrors( status );

  // compute and print the performance
  
  double secondsTotal = 0.001 * (double)msecTotal;
  double megaTrialsPerSecond =   (float)NUMTRIALS / secondsTotal / 1000000.;
  // double megaMultsPerSecond = multsPerSecond / 1000000.;
  fprintf( stderr, "Number of Trials = %10d, Block Size = %10d, MegaTrials/Second = %10.2lf\n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond );
  fprintf(f,"%d  %d  %f  %f \n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond, 1.0 );
  // copy result from the device to the host:

  status = cudaMemcpy( hT, dT, NUMTRIALS*sizeof(float), cudaMemcpyDeviceToHost );
    checkCudaErrors( status );

          // // check for correctness:
          // for(int i = 1; i < NUMTRIALS; i++ )
          // {
          // double error = ( (double)hrs[ i ] - (double)i ) / (double)i;
          // if( fabs(error) > TOLERANCE )
          // {
          // fprintf( stderr, "C[%10d] = %10.2lf, correct = %10.2lf\n",
          // i, (double)hrs[ i ], (double)i );
          // }
          // }

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
