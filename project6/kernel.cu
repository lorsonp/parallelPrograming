// array multiplication (CUDA Kernel) on the device: C = A * B
__global__ void ArrayMul( float *A, float *B, float *C )
{
int gid = blockIdx.x*blockDim.x + threadIdx.x;
C[gid] = A[gid] * B[gid];
}

__global__ void LaserMonteCarlo( float *XCS, float *YCS, float *RS, float *T )
{
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

      if (d>0) {
      // hits the circle:
      // get the first intersection:
      d = sqrt( d );
      float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
      float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
      float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

          if (tmin>0) {

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
          float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
          float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

          // find out if it hits the infinite plate:
          float t_other = ( 0. - ycir ) / outy;

              //IF  t is less than 0., then the reflected beam went up instead of down.
              //Continue on to the next trial in the for-loop.
                  //Otherwise, this beam hit the infinite plate. (Case D) Increment the number
                  //of hits and continue on to the next trial in the for-loop.
                  if (t_other>0) {
                    T += 1;
                  }
  }
  }


}

__global__ void LaserMonteCarloNoIf( float *XCS, float *YCS, float *RS, float *T )
{
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
            float nx = xcs[gid]ir - xcs[gid];
            float ny = ycir - ycs[gid];
            float n = sqrt( nx*nx + ny*ny );
            nx /= n;	// unit vector
            ny /= n;	// unit vector

            // get the unitized incoming vector:
            float inx = xcs[gid]ir - 0.;
            float iny = ycir - 0.;
            float in = sqrt( inx*inx + iny*iny );
            inx /= in;	// unit vector
            iny /= in;	// unit vector

            // get the outgoing (bounced) vector:
            float dot = inx*nx + iny*ny;
            float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
            float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

            // find out if it hits the infinite plate:
            float t = ( 0. - ycir ) / outy;
            if (t > 0)
          		{
          			T[gid] = 1;
          		}
            else
              {
                T[gid] = 0;
              }
            // for (int offset = 1; offset < numItems; offset *= 2)
          	// {
            //   __syncthreads();
          	// 	if (t > 0)
          	// 	{
          	// 		numHits += 1;
          	// 	}
          	// }

                //IF  t is less than 0., then the reflected beam went up instead of down.
                //Continue on to the next trial in the for-loop.
                //Otherwise, this beam hit the infinite plate. (Case D) Increment the number
                //of hits and continue on to the next trial in the for-loop.

}
