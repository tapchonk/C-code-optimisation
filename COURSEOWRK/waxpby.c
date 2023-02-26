#undef _GLIBCXX_DEBUG  

#include <immintrin.h>
#include <pthread.h>

#pragma GCC optimize("O3,inline")
#pragma GCC target("bmi,bmi2,lzcnt,popcnt")

#include "waxpby.h"

#define MAX_THREAD 4

__m256 alphaVector;
__m256 betaVector;

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors. Now with parallelisation and SIMD vector intrinsics!
 *  
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */

int MAXWAXPBY;
int partWAXPBY, loopFactorWAXPBY = 64;
pthread_mutex_t lockWAXPBY;

struct argStruct {
  const float *arrX;
  const float *arrY;
  const float *arrW;
  double alpha;
  double beta;
};

void* alphaWAXPBYProcess(void* arg1) {
  pthread_mutex_lock(&lockWAXPBY);
  register int threadPart = partWAXPBY++;
  pthread_mutex_unlock(&lockWAXPBY);
  register int i, start = threadPart * (MAXWAXPBY/MAX_THREAD), end = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/loopFactorWAXPBY)*loopFactorWAXPBY + start;
  register int endEight = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/8)*8 + start;
  //printf("this is thread: %d\n", threadPart);
  register struct argStruct *args = arg1;
  register float *x = (float *)(args->arrX);
  register float *y = (float *)(args->arrY);
  register float *w = (float *)(args->arrW);
  register double beta = (double)(args->beta);
  for (i=start; i<end; i+=loopFactorWAXPBY) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_loadu_ps(x + i     ), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i     ))));
    _mm256_storeu_ps(w + i + 8 , _mm256_add_ps(_mm256_loadu_ps(x + i +  8), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i +  8))));
    _mm256_storeu_ps(w + i + 16, _mm256_add_ps(_mm256_loadu_ps(x + i + 16), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 16))));
    _mm256_storeu_ps(w + i + 24, _mm256_add_ps(_mm256_loadu_ps(x + i + 24), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 24))));
    _mm256_storeu_ps(w + i + 32, _mm256_add_ps(_mm256_loadu_ps(x + i + 32), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 32))));
    _mm256_storeu_ps(w + i + 40, _mm256_add_ps(_mm256_loadu_ps(x + i + 40), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 40))));
    _mm256_storeu_ps(w + i + 48, _mm256_add_ps(_mm256_loadu_ps(x + i + 48), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 48))));
    _mm256_storeu_ps(w + i + 56, _mm256_add_ps(_mm256_loadu_ps(x + i + 56), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i + 56))));
  }
  for (; i<endEight; i+=8) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_loadu_ps(x + i     ), _mm256_mul_ps(betaVector, _mm256_loadu_ps(y + i     ))));
  }
  for (; i<(threadPart + 1) * (MAXWAXPBY/MAX_THREAD); i++) {
    w[i] = x[i] + beta * y[i];
  }
  return 0;
}

void* betaWAXPBYProcess(void* arg1) {
  pthread_mutex_lock(&lockWAXPBY);
  register int threadPart = partWAXPBY++;
  pthread_mutex_unlock(&lockWAXPBY);
  register int i, start = threadPart * (MAXWAXPBY/MAX_THREAD), end = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/loopFactorWAXPBY)*loopFactorWAXPBY + start;
  register int endEight = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/8)*8 + start;
  //printf("this is thread: %d\n", threadPart);
  register struct argStruct *args = arg1;
  register float *x = (float *)(args->arrX);
  register float *y = (float *)(args->arrY);
  register float *w = (float *)(args->arrW);
  register double alpha = (double)(args->alpha);
  for (i=start; i<end; i+=loopFactorWAXPBY) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_loadu_ps(y + i    ), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i    ))));
    _mm256_storeu_ps(w + i + 8 , _mm256_add_ps(_mm256_loadu_ps(y + i+ 8 ), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 8 ))));
    _mm256_storeu_ps(w + i + 16, _mm256_add_ps(_mm256_loadu_ps(y + i+ 16), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 16))));
    _mm256_storeu_ps(w + i + 24, _mm256_add_ps(_mm256_loadu_ps(y + i+ 24), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 24))));
    _mm256_storeu_ps(w + i + 32, _mm256_add_ps(_mm256_loadu_ps(y + i+ 32), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 32))));
    _mm256_storeu_ps(w + i + 40, _mm256_add_ps(_mm256_loadu_ps(y + i+ 40), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 40))));
    _mm256_storeu_ps(w + i + 48, _mm256_add_ps(_mm256_loadu_ps(y + i+ 48), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 48))));
    _mm256_storeu_ps(w + i + 56, _mm256_add_ps(_mm256_loadu_ps(y + i+ 56), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 56))));
  }
  for (; i<endEight; i+=8) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_loadu_ps(y + i    ), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i    ))));
  }
  for (; i<(threadPart + 1) * (MAXWAXPBY/MAX_THREAD); i++) {
    w[i] = alpha * x[i] + y[i];
  }
  return 0;
}

void* elseWAXPBYProcess(void* arg1) {
  pthread_mutex_lock(&lockWAXPBY);
  register int threadPart = partWAXPBY++;
  pthread_mutex_unlock(&lockWAXPBY);
  register int i, start = threadPart * (MAXWAXPBY/MAX_THREAD), end = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/loopFactorWAXPBY)*loopFactorWAXPBY + start;
  register int endEight = (((threadPart + 1) * (MAXWAXPBY/MAX_THREAD)  - start)/8)*8 + start;
  //printf("this is thread: %d\n", threadPart);
  register struct argStruct *args = arg1;
  register float *x = (float *)(args->arrX);
  register float *y = (float *)(args->arrY);
  register float *w = (float *)(args->arrW);
  register double alpha = (double)(args->alpha);
  register double beta = (double)(args->beta);
  for (i=start; i<end; i+=loopFactorWAXPBY) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i    ), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i    ))));
    _mm256_storeu_ps(w + i + 8 , _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 8 ), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 8 ))));
    _mm256_storeu_ps(w + i + 16, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 16), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 16))));
    _mm256_storeu_ps(w + i + 24, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 24), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 24))));
    _mm256_storeu_ps(w + i + 32, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 32), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 32))));
    _mm256_storeu_ps(w + i + 40, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 40), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 40))));
    _mm256_storeu_ps(w + i + 48, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 48), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 48))));
    _mm256_storeu_ps(w + i + 56, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i+ 56), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i+ 56))));
  }
  for (; i<endEight; i+=8) {
    _mm256_storeu_ps(w + i     , _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(y + i    ), betaVector), _mm256_mul_ps(alphaVector, _mm256_loadu_ps(x + i    ))));
  }
  for (; i<(threadPart + 1) * (MAXWAXPBY/MAX_THREAD); i++) {
    w[i] = alpha * x[i] + beta * y[i];
  }
  return 0;
} 

__inline__ int waxpby(const int n, const double alpha, const float * const x, const double beta, const float * const y,  float * const w) {
  register int i;
  struct argStruct args;
  args.arrX = x;
  args.arrY = y;
  args.arrW = w;
  args.alpha = alpha;
  args.beta  = beta;
  MAXWAXPBY = n;
  partWAXPBY = 0;
  alphaVector = _mm256_set1_ps(alpha);
  betaVector  = _mm256_set1_ps(beta);
  pthread_t threads[MAX_THREAD];

  if (pthread_mutex_init(&lockWAXPBY, NULL) != 0) {
    return 1;
  }
  
  if (__builtin_expect(alpha == 1.0, 1)){
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, alphaWAXPBYProcess, (void*)&args);
      }
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_join(threads[i], NULL);
      }
  } else if (__builtin_expect(beta == 1.0, 1)){
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, betaWAXPBYProcess, (void*)&args);
      }
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_join(threads[i], NULL);
      }
  } else {
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, elseWAXPBYProcess, (void*)&args);
      }
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_join(threads[i], NULL);
      }
  }

  if (n % MAX_THREAD != 0) {
    if (__builtin_expect(alpha == 1.0, 0)) {
      for (i=(n/MAX_THREAD)*MAX_THREAD; i<n; i++) {
        w[i] = x[i] + beta * y[i];
      }
    } else if(__builtin_expect(beta == 1.0, 0)) {
      for (i=(n/MAX_THREAD)*MAX_THREAD; i<n; i++) {
          w[i] = alpha * x[i] + y[i];
        }
    } else {
      for (i=(n/MAX_THREAD)*MAX_THREAD; i<n; i++) {
        w[i] = alpha * x[i] + beta * y[i];
      }
    }
  }
  pthread_mutex_destroy(&lockWAXPBY);
  partWAXPBY = 0;
  
  return 0;
}