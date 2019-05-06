

float
SQR( float x )
{
        return x*x;
}
int main(){

int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population

const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =		0.5;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

// starting date and time:
NowMonth =    0;
NowYear  = 2019;

// starting state (feel free to change this if you want):
NowNumDeer = 1;
NowHeight =  1.;

omp_set_num_threads( 4 );	// same as # of sections
#pragma omp parallel sections
{
	#pragma omp section
	{
		// GrainDeer( );

			if (NowHeight>NowNumDeer) {
				NewNumDeer = NowNumDeer - 1;
		} else if (NowHeight<NowNumDeer) {
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

	#pragma omp section
	{
		while( NowYear < 2025 ){
				// Grain( );
				float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
		    float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

				NewHeight = NowHeight + tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
				NewHeight = NowHeight - (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
				if (NowHeight<0) {
					NowHeight = 0;
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

	#pragma omp section
	{
		while( NowYear < 2025 ){
				// Watcher( );

				// DoneComputing barrier:
				#pragma omp barrier

				// DoneAssigning barrier:
				#pragma omp barrier
				float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

				float temp = AVG_TEMP - AMP_TEMP * cos( ang );
				unsigned int seed = 0;
				NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

				float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
				NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
				if( NowPrecip < 0. )
					NowPrecip = 0.;
			}
	}

	#pragma omp section
	{
		MyAgent( );	// your own
	}
}       // implied barrier -- all functions must return in order
	// to allow any of them to get past here
}
