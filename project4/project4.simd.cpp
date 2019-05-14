#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "simd.p4.h"


#ifndef ARRAYSIZE
#define ARRAYSIZE       1000000	// you decide
#endif

#ifndef NUMTRIES
#define NUMTRIES       5	// you decide
#endif

float *A = new float [ARRAYSIZE];
float *B = new float [ARRAYSIZE];
float *C = new float [ARRAYSIZE];

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
FILE *f1;
FILE *f2;
FILE *f3;
FILE *f4;
f1 = fopen("project4.simd.Mult.txt","a");
f2 = fopen("project4.simd.sumMult.txt","a");
f3 = fopen("project4.nonsimd.Mult.txt","a");
f4 = fopen("project4.nonsimd.sumMult.txt","a");


    double maxPerformance = 0.;

    for( int t = 0; t < NUMTRIES; t++ )
    {
            double time0 = omp_get_wtime( );

            SimdMul( A, B, C, ARRAYSIZE );

            double time1 = omp_get_wtime( );

						double megaCalcs = (double)ARRAYSIZE / ( time1 - time0 ) / 1000000.;
						if( megaCalcs > maxPerformance )
							maxPerformance = megaCalcs;

            // double milliSecs = (time1-time0)*1000.;
            // if( milliSecs < minMilliSecs )
            //         minMilliSecs = milliSecs;
    }

    fprintf(f1,"%f \n", maxPerformance);

    maxPerformance = 0.;

    for( int t = 0; t < NUMTRIES; t++ )
    {
            double time0 = omp_get_wtime( );

            SimdMulSum( A, B, ARRAYSIZE );

            double time1 = omp_get_wtime( );

						double megaCalcs = (double)ARRAYSIZE / ( time1 - time0 ) / 1000000.;
						if( megaCalcs > maxPerformance )
							maxPerformance = megaCalcs;

            // double milliSecs = (time1-time0)*1000.;
            // if( milliSecs < minMilliSecs )
            //         minMilliSecs = milliSecs;
    }

    fprintf(f2,"%f \n", maxPerformance);

    return 0;
}
