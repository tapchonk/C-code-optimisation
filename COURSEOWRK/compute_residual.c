#include <math.h>  // needed for fabs
#include "compute_residual.h"
#include <immintrin.h>
#include <pthread.h>

#pragma GCC optimize("O3,inline")
#pragma GCC target("bmi,bmi2,lzcnt,popcnt")

#define MAX_THREAD 4

/**
 * @brief Compute the 1-norm difference between two vectors. Now with parallelisation and SIMD vector intrinsics!
 *
 * @param n Number of vector elements
 * @param v1 Input vector
 * @param v2 Input vector
 * @param residual Pointer to scalar return value
 * @return int 0 if no error
 */

double maxResidualArray[MAX_THREAD];
int MAXRESIDUAL;
int partRESIDUAL, loopFactorRESIDUAL = 4;
pthread_mutex_t lockRESIDUAL;

struct argStruct {
  const float *arrV1;
  const double *arrV2;
};

void* maxResidualProcess(void* arg1) {
  pthread_mutex_lock(&lockRESIDUAL);
  register int threadPart = partRESIDUAL++;
  pthread_mutex_unlock(&lockRESIDUAL);
  register int i, start = threadPart * (MAXRESIDUAL/MAX_THREAD), end = (((threadPart + 1) * (MAXRESIDUAL/MAX_THREAD)  - start)/loopFactorRESIDUAL)*loopFactorRESIDUAL + start;
  register struct argStruct *args = arg1;
  register float  *v1 = (float  *)(args->arrV1);
  register double *v2 = (double *)(args->arrV2);
  register double local_residual = 0.0;
  __m256d maxVec = _mm256_set1_pd(0.0);
  __m256d v1Vec;   
  __m256d v2Vec;     
  __m256d subVec;    
  __m256d signMask;  
  __m256d subVecAbs; 
  __m256d mask;      
  for (i=start; i<end; i+=loopFactorRESIDUAL) {
    v1Vec     = _mm256_cvtps_pd(_mm_loadu_ps(v1));
    v2Vec     = _mm256_loadu_pd(v2);
    subVec    = _mm256_sub_pd(v2Vec, v1Vec);
    signMask  = _mm256_set1_pd(-0.);
    subVecAbs = _mm256_andnot_pd(signMask, subVec);
    mask      = _mm256_cmp_pd(maxVec, subVecAbs, _CMP_GT_OQ);
    maxVec = _mm256_blendv_pd(subVecAbs, maxVec, mask);
  }
  double * maxFour = _mm_malloc(4 * sizeof(double), 64);
  _mm256_store_pd(maxFour, maxVec);
  for (i=0; i<loopFactorRESIDUAL; i++){
    if(maxFour[i] > local_residual) {
      local_residual = maxFour[i];
    }
  }
  for (; i<(threadPart + 1) * (MAXRESIDUAL/MAX_THREAD); i++){
    double diff = fabs((double)v1[i] - v2[i]);
    if (diff > local_residual) {
      local_residual = diff;
    }
  }

  pthread_mutex_lock(&lockRESIDUAL);
  maxResidualArray[threadPart] = local_residual;
  pthread_mutex_unlock(&lockRESIDUAL);
  return 0;
}

__inline__ int compute_residual(const int n, const float * const v1, const double * const v2, double * const residual)
{
  struct argStruct args;
  partRESIDUAL = 0;
  args.arrV1 = v1;
  args.arrV2 = v2;
  MAXRESIDUAL = n;
  int i;
  pthread_t threads[MAX_THREAD];

  for(i = 0; i < MAX_THREAD; i++) {
    maxResidualArray[i] = 0.0;
  }

  if (pthread_mutex_init(&lockRESIDUAL, NULL) != 0) {
    return 1;
  }

  for (i = 0; i < MAX_THREAD; i++) {
    pthread_create(&threads[i], NULL, maxResidualProcess, (void*)&args);
  }

  for (i = 0; i < MAX_THREAD; i++) {
    pthread_join(threads[i], NULL);
  }

  double local_residual = 0.0;

  for (i = 0; i < MAX_THREAD; i++) {
    if(local_residual < maxResidualArray[i]) {
      local_residual = maxResidualArray[i];
    }
  }

  if ((MAXRESIDUAL % MAX_THREAD) != 0) {
    for (i=(MAXRESIDUAL/MAX_THREAD)*MAX_THREAD; i<n; i++) {
      double diff = fabs((double)v1[i] - v2[i]);
      if (diff > local_residual) {
        local_residual = diff;
      }
    }
  }

  *residual = local_residual;

  pthread_mutex_destroy(&lockRESIDUAL);
  partRESIDUAL = 0;
  for(i = 0; i < MAX_THREAD; i++) {
    maxResidualArray[i] = 0.0;
  }

  return 0;
}
