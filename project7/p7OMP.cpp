#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>

// setting the number of threads:
#ifndef NUMT
#define NUMT		1
#endif

// setting the number of tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

int
main (int argc, char const *argv[])
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


  omp_set_num_threads( NUMT );
  fprintf( stderr, "Using %d threads\n", NUMT );

  double maxMegaCalcs = 0.;

  for( int t = 0; t < NUMTRIES; t++ )
  {
          double time0 = omp_get_wtime( );

          #pragma omp parallel for
          for( int shift = 0; shift < Size; shift++ )
          {
            float sum = 0.;
            for( int i = 0; i < Size; i++ )
            {
              sum += Array[i] * Array[i + shift];
            }
            Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
          }
          double time1 = omp_get_wtime( );
          double megaCalcs = Size/(time1-time0)/1000000.;
          if( megaCalcs > maxMegaCalcs )
                  maxMegaCalcs = megaCalcs;
   }

    FILE *f;
    f = fopen("SumsVShift.txt","a");
    if (NUMT == 1)
    {
      fprintf(f,"\n")
      for (size_t i = 1; i < 512; i++)
      {
      fprintf(f,"%f  %d \n", Sums[i], i)
      }
    }

    FILE *f;
    f = fopen("p7OMP.txt","a");
    fprintf(f,"%d  %d  %f  %f \n", NUMTRIALS, NUMT, maxPerformance, currentProb);

return 0;
}
