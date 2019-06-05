#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "simd.p4.h"


#ifndef NUMTRIES
#define NUMTRIES       5	// you decide
#endif

using namespace std;


void
SimdMul( float *a, float *b,   float *c,   int len )
{
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	__asm
	(
		".att_syntax\n\t"
		"movq    -24(%rbp), %r8\n\t"		// a
		"movq    -32(%rbp), %rcx\n\t"		// b
		"movq    -40(%rbp), %rdx\n\t"		// c
	);

	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%r8), %xmm0\n\t"	// load the first sse register
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply
			"movups	%xmm0, (%rdx)\n\t"	// store the result
			"addq $16, %r8\n\t"
			"addq $16, %rcx\n\t"
			"addq $16, %rdx\n\t"
		);
	}

	for( int i = limit; i < len; i++ )
	{
		c[i] = a[i] * b[i];
	}
}

float
SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;

	__asm
	(
		".att_syntax\n\t"
		"movq    -40(%rbp), %r8\n\t"		// a
		"movq    -48(%rbp), %rcx\n\t"		// b
		"leaq    -32(%rbp), %rdx\n\t"		// &sum[0]
		"movups	 (%rdx), %xmm2\n\t"		// 4 copies of 0. in xmm2
	);

	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%r8), %xmm0\n\t"	// load the first sse register
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply
			"addps	%xmm0, %xmm2\n\t"	// do the add
			"addq $16, %r8\n\t"
			"addq $16, %rcx\n\t"
		);
	}
}

float
NonSimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;

	for( int i = 0; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0];
}



int
main( )
{
  #ifndef _OPENMP
          fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
          return 1;
  #endif



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



  double maxPerformance = 0.;

  for( int t = 0; t < NUMTRIES; t++ )
  {
  double time0 = omp_get_wtime( );
  for( int shift = 0; shift < Size; shift++ )
  {
      Sums[shift] = SimdMulSum( &Array[0], &Array[0+shift], Size );
  }
  double time1 = omp_get_wtime( );

  double megaCalcs = Size / ( time1 - time0 ) / 1000000.;
  if( megaCalcs > maxPerformance )
    maxPerformance = megaCalcs;
  }

  FILE *f;
  f = fopen("SumsVShift.txt","a");
  fprintf(f,"\n")
  for (size_t i = 1; i < 512; i++)
  {
  fprintf(f,"%f  %d \n", Sums[i], i)
  }

  FILE *f;
  f = fopen("p7SIMD.txt","a");
  fprintf(f," %f \n", maxPerformance);

return 0;
}
