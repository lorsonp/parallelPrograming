#include <omp.h>
#include <stdio.h>
#include <math.h>

#define ARRAYSIZE       1000000	// you decide
#define NUMTRIES        100	// you decide
#define N               4

int
main( )
{
#ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
#endif
// float N[] = {1, 4};
// double maxMegaMults[2] = {0., 0.};
// for( int j = 0; j < 2; j++ )
// {
        float A[ARRAYSIZE];
        float B[ARRAYSIZE];
        float C[ARRAYSIZE];

        // #define NUMT    N[j]
        omp_set_num_threads( N );
        fprintf( stderr, "Using %d threads\n", N );

        double maxMegaMults = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                #pragma omp parallel for
                for( int i = 0; i < ARRAYSIZE; i++ )
                {
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime( );
                double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.;
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
        }

        printf( "Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults );
// }
// double S = maxMegaMults[0]/maxMegaMults[1];
// printf( "Speedup = %8.3lf \n", S );
//
// double Fp = (4./3.)*( 1. - (1./S) );
// printf( "Parallel Fraction = %8.3lf\n", Fp );
	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

        return 0;
}
