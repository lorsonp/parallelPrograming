#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>

int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11
int	NowNumDeer;		// number of deer in the current population
float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
float Income;

const float GRAIN_GROWS_PER_MONTH =		9.0;
const float ONE_DEER_EATS_PER_MONTH =		0.3;
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
Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;
        return (int)(  Ranf(seedp, low,high) );
}

float
SQR( float x )
{
        return x*x;
}

void GrainDeer()
{
  int NewNumDeer;
  while(NowYear<2025)
  {
    if ((float)NowHeight<(float)NowNumDeer)
    {
      NewNumDeer = NowNumDeer - 1;
    }
    else
    {
      NewNumDeer = NowNumDeer + 1;
    }
    if ((float)NowHeight<(float)NowNumDeer) {
      NewNumDeer = NowNumDeer - 1;
    } else if ((float)NowHeight>(float)NowNumDeer) {
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
}

void Grain()
{
  float	NewHeight;		// grain height in inches

  while( NowYear < 2025 ){
      // Grain( );
      float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
      printf("%f \n",ang);
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
      NewHeight = NewHeight - (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
      if  (NowHeight>HARVEST_HEIGHT) {
        Income += (NowHeight - HARVESTED_HEIGHT)*COST_OF_GRAIN_PER_INCH;
        NewHeight = HARVESTED_HEIGHT;
      }
      if (NewHeight<=0) {
        NewHeight = 0;
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
  FILE *f;
  f = fopen("project3.txt","a");

  while( NowYear < 2025 ){
      // Watcher( );
      printf("Computing \n");
      // DoneComputing barrier:
      #pragma omp barrier
      printf("Done Computing \n");
      // DoneAssigning barrier:
      #pragma omp barrier
      printf("Done Assigning \n");
      fprintf(f,"%d  %d  %f  %f  %f  %d  %f \n", NowYear, NowMonth, NowTemp, NowPrecip, NowHeight, NowNumDeer, Income);
      NowMonth += 1;
      printf("%d \n",NowMonth);
      if (NowMonth==12) {
        printf("here\n");
        NowYear+=1;
        NowMonth = 0;
       }
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
  Income = 0;			// income based on the harvested grain height
	// starting state (feel free to change this if you want):
	NowNumDeer = 1;
	NowHeight =  1.;

	omp_set_num_threads( 3 );	// same as # of sections
	#pragma omp parallel sections default(none)
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

	}
}
