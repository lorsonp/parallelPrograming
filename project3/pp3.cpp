#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>

int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11
int	NowNumDeer;		// number of deer in the current population
int NewNumDeer;

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
float	NewHeight;		// grain height in inches
float Income;			// income based on the harvested grain height

const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =		0.5;
const float HARVESTED_HEIGHT = 8.0;
const float HARVEST_HEIGHT =  24.0;
const float COST_OF_GRAIN_PER_INCH =  9;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;


float
SQR( float x )
{
        return x*x;
}

int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

void GrainDeer()
{
    if (NowHeight>NowNumDeer)
    {
      NewNumDeer = NowNumDeer - 1;
  }
  else if (NowHeight<NowNumDeer)
  {
      NewNumDeer = NowNumDeer + 1;
  } else {
      NewNumDeer = NowNumDeer;
  }
  // DoneComputing barrier:
  #pragma omp barrier
  NowNumDeer = NewNumDeer;

  // DoneAssigning barrier:
  #pragma omp barrier

  // DonePrinting barrier:
  #pragma omp barrier
}

void Grain()
{
  while( NowYear < 2025 ){
      // Grain( );
      float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
      float temp = AVG_TEMP - AMP_TEMP * cos( ang );
      unsigned int seed = 0;
      NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

      float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
      NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
      if( NowPrecip < 0. )
        {NowPrecip = 0.;}

      float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
      float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

      NewHeight = NowHeight + tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
      NewHeight = NowHeight - (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
      if  (NowHeight>HARVEST_HEIGHT) {
        float NewIncome = Income + (NowHeight - HARVESTED_HEIGHT)*COST_OF_GRAIN_PER_INCH;
        NewHeight = HARVESTED_HEIGHT;
      }
      if (NowHeight<0) {
        NewHeight = 0;
        Income = NewIncome
      }

      // DoneComputing barrier:
      #pragma omp barrier
      NowHeight = NewHeight;

      // DoneAssigning barrier:
      #pragma omp barrier

      // DonePrinting barrier:
      #pragma omp barrier
    }
}

void Watcher()
{
  while( NowYear < 2025 ){
      // Watcher( );
      printf("Computing");
      // DoneComputing barrier:
      #pragma omp barrier
      printf("Done Computing");
      // DoneAssigning barrier:
      #pragma omp barrier
      printf("Done Assigning");
      // FILE *f;
      // f = fopen("project3.txt","a");
      // fprintf(f,"%d  %d  %f  %f  %f  %d  %f \n", NowYear, NowMonth, NowTemp, NowPrecip, NowHeight, NowNumDeer, Income);
      NowYear+=1;
      // DonePrinting barrier:
      #pragma omp barrier
    }
}

int main(){

	#ifndef _OPENMP
		fprintf( stderr, "No OpenMP support!\n" );
		return 1;
	#endif

	// starting date and time:
	NowMonth =    0;
	NowYear  = 2019;
	Income = 0;
	// starting state (feel free to change this if you want):
	NowNumDeer = 1;
	NowHeight =  1.;

	omp_set_num_threads( 3 );	// same as # of sections
	#pragma omp parallel sections
	{
		#pragma omp section
		{

			GrainDeer( );

		}

		#pragma omp section
		{
      Grain();
		}

		#pragma omp section
		{
			Watcher();
		}

		// 	#pragma omp section
		// 	{
		// 		while( NowYear < 2025 ){
		// 			// MyAgent( );	// Harvest
    //
		// 			// DoneComputing barrier:
		// 			#pragma omp barrier
    //
		// 			// DoneAssigning barrier:
		// 			#pragma omp barrier
    //
		// 			// DonePrinting barrier:
		// 			#pragma omp barrier
		// 		}
		// }
	}       // implied barrier -- all functions must return in order
					// to allow any of them to get past here
	return 0;
}
