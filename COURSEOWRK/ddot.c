#undef _GLIBCXX_DEBUG  

#include "ddot.h"
#include <immintrin.h>
#include <pthread.h>

#pragma GCC optimize("O3,inline")
#pragma GCC target("bmi,bmi2,lzcnt,popcnt")

#define MAX_THREAD 4

/**
 * @brief Compute the dot product of two vectors. Now with parallelisation and SIMD vector intrinsics!
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */

double sumDDOTArray[MAX_THREAD];
int MAXDDOT;
int partDDOT, loopFactorDDOT = 128;
pthread_mutex_t lockDDOT;

struct argStruct {
  const float *arrX;
  const float *arrY;
};

void* sumXXArray(void* arg1) {
  pthread_mutex_lock(&lockDDOT);
  register int threadPart = partDDOT++;
  pthread_mutex_unlock(&lockDDOT);
  register int i, start = threadPart * (MAXDDOT/MAX_THREAD), end = (((threadPart + 1) * (MAXDDOT/MAX_THREAD)  - start)/loopFactorDDOT)*loopFactorDDOT + start;
  register int end32 = ((((threadPart + 1) * (MAXDDOT/MAX_THREAD)  - start)>>5)<<5) + start;
  register float *x = (float*)arg1;
  register double tempSum = 0.0;  
  register __m128 hiQuadSum,      
                  hiDualSum,      
                  singleSum;
  register __m256 xVecSquare,
                  xVecSquareSum1, 
                  xVecSquareSum2, 
                  xVecSquareSum3, 
                  xVecSquareSum4,    
                  xVec1, 
                  xVec2, 
                  xVec3, 
                  xVec4, 
                  xVec5, 
                  xVec6, 
                  xVec7, 
                  xVec8, 
                  xVec9, 
                  xVec10,
                  xVec11,
                  xVec12,
                  xVec13,
                  xVec14,
                  xVec15,
                  xVec16;    

  for (i=start ; i<end; i+=loopFactorDDOT) {
      xVec1   = _mm256_loadu_ps(x + i     );
      xVec2   = _mm256_loadu_ps(x + i + 8 );
      xVec3   = _mm256_loadu_ps(x + i + 16);
      xVec4   = _mm256_loadu_ps(x + i + 24);
      xVec5   = _mm256_loadu_ps(x + i + 32);
      xVec6   = _mm256_loadu_ps(x + i + 40);
      xVec7   = _mm256_loadu_ps(x + i + 48);
      xVec8   = _mm256_loadu_ps(x + i + 56);
      xVec9   = _mm256_loadu_ps(x + i + 64);
      xVec10  = _mm256_loadu_ps(x + i + 72);
      xVec11  = _mm256_loadu_ps(x + i + 80);
      xVec12  = _mm256_loadu_ps(x + i + 88);
      xVec13  = _mm256_loadu_ps(x + i + 96);
      xVec14  = _mm256_loadu_ps(x + i + 104);
      xVec15  = _mm256_loadu_ps(x + i + 112);
      xVec16  = _mm256_loadu_ps(x + i + 120);
      xVecSquareSum1 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec1 , xVec1  ),_mm256_mul_ps(xVec2 , xVec2   )),_mm256_mul_ps(xVec3 , xVec3  )),_mm256_mul_ps(xVec4 , xVec4   ));
      xVecSquareSum2 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec5 , xVec5  ),_mm256_mul_ps(xVec6 , xVec6   )),_mm256_mul_ps(xVec7 , xVec7  )),_mm256_mul_ps(xVec8 , xVec8   ));
      xVecSquareSum3 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec9 , xVec9  ),_mm256_mul_ps(xVec10 , xVec10 )),_mm256_mul_ps(xVec11 , xVec11)),_mm256_mul_ps(xVec12 , xVec12 ));
      xVecSquareSum4 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec13, xVec13 ),_mm256_mul_ps(xVec14 , xVec14 )),_mm256_mul_ps(xVec15 , xVec15)),_mm256_mul_ps(xVec16 , xVec16 ));
      xVecSquare     = _mm256_add_ps(_mm256_add_ps(xVecSquareSum1, xVecSquareSum2),_mm256_add_ps(xVecSquareSum3,xVecSquareSum4));
      hiQuadSum      = _mm_add_ps(_mm256_castps256_ps128(xVecSquare), _mm256_extractf128_ps(xVecSquare, 1));
      hiDualSum      = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
      singleSum      = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));

        tempSum = tempSum + _mm_cvtss_f32(singleSum);
  }

  for (; i<end32; i+=32) {
      xVec1  = _mm256_loadu_ps(x + i     );
      xVec2  = _mm256_loadu_ps(x + i +  8);
      xVec3  = _mm256_loadu_ps(x + i + 16);
      xVec4  = _mm256_loadu_ps(x + i + 24);
      xVecSquareSum1 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec1 , xVec1 ),_mm256_mul_ps(xVec2 , xVec2 )),_mm256_mul_ps(xVec3 , xVec3 )),_mm256_mul_ps(xVec4 , xVec4 ));
      hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(xVecSquareSum1), _mm256_extractf128_ps(xVecSquareSum1, 1));
      hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
      singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));

        tempSum = tempSum + _mm_cvtss_f32(singleSum);
  }

  for (; i<(threadPart + 1) * (MAXDDOT / MAX_THREAD); i++) {
    tempSum += x[i]*x[i];
  }

  pthread_mutex_lock(&lockDDOT);
  sumDDOTArray[threadPart] = tempSum;
  pthread_mutex_unlock(&lockDDOT);
  return 0;
} 

void* sumXYArray(void* arg1) {
  pthread_mutex_lock(&lockDDOT);
  register int threadPart = partDDOT++;
  pthread_mutex_unlock(&lockDDOT);
  register int i, start = threadPart * (MAXDDOT/MAX_THREAD), end = (((threadPart + 1) * (MAXDDOT/MAX_THREAD)  - start)/loopFactorDDOT)*loopFactorDDOT + start;
  register int end32 = ((((threadPart + 1) * (MAXDDOT/MAX_THREAD) - start)>>5)<<5) + start;
  //printf("this is thread: %d\n", threadPart);
  register struct argStruct *args = arg1;
  register float *x = (float *)(args->arrX);
  register float *y = (float *)(args->arrY);
  register __m256 xVecSquareSum1, 
                  xVecSquareSum2, 
                  xVecSquareSum3, 
                  xVecSquareSum4, 
                  xVecSquare;     
  register __m128 hiQuadSum,      
                  hiDualSum,      
                  singleSum;      

  double tempSum = 0.0;
  
  for (i=start ; i<end; i+=loopFactorDDOT) {
    xVecSquareSum1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i     ),_mm256_loadu_ps(y + i     )),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 8  ), _mm256_loadu_ps(y + i + 8  ))),_mm256_mul_ps(_mm256_loadu_ps(x + i+ 16  ), _mm256_loadu_ps(y + i+ 16  ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 24 ), _mm256_loadu_ps(y + i + 24 )));
    xVecSquareSum2 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 32),_mm256_loadu_ps(y + i + 32)),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 40 ), _mm256_loadu_ps(y + i + 40 ))),_mm256_mul_ps(_mm256_loadu_ps(x + i + 48 ), _mm256_loadu_ps(y + i + 48 ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 56 ), _mm256_loadu_ps(y + i + 56 )));
    xVecSquareSum3 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 64),_mm256_loadu_ps(y + i + 64)),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 72 ), _mm256_loadu_ps(y + i + 72 ))),_mm256_mul_ps(_mm256_loadu_ps(x + i+ 80  ), _mm256_loadu_ps(y + i+ 80  ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 88 ), _mm256_loadu_ps(y + i + 88 )));
    xVecSquareSum4 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 96),_mm256_loadu_ps(y + i + 96)),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 104), _mm256_loadu_ps(y + i + 104))),_mm256_mul_ps(_mm256_loadu_ps(x + i + 112), _mm256_loadu_ps(y + i + 112))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 120), _mm256_loadu_ps(y + i + 120)));
    xVecSquare     = _mm256_add_ps(_mm256_add_ps(xVecSquareSum1, xVecSquareSum2),_mm256_add_ps(xVecSquareSum3, xVecSquareSum4));
    hiQuadSum      = _mm_add_ps(_mm256_castps256_ps128(xVecSquare), _mm256_extractf128_ps(xVecSquare, 1));
    hiDualSum      = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
    singleSum      = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
    tempSum = tempSum + _mm_cvtss_f32(singleSum);
  }

  for (; i<end32; i+=32) {
    xVecSquareSum1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i     ),_mm256_loadu_ps(y + i     )),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 8  ), _mm256_loadu_ps(y + i + 8  ))),_mm256_mul_ps(_mm256_loadu_ps(x + i+ 16  ), _mm256_loadu_ps(y + i+ 16  ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 24 ), _mm256_loadu_ps(y + i + 24 )));
    hiQuadSum = _mm_add_ps(_mm256_castps256_ps128(xVecSquareSum1), _mm256_extractf128_ps(xVecSquareSum1, 1));
    hiDualSum = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
    singleSum = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
    tempSum = tempSum + _mm_cvtss_f32(singleSum);
  }
  for (; i<(threadPart + 1) * (MAXDDOT / MAX_THREAD); i++) {
    tempSum += x[i]*y[i];
  }

  pthread_mutex_lock(&lockDDOT);
  sumDDOTArray[threadPart] = tempSum;
  pthread_mutex_unlock(&lockDDOT);
  return 0;
} 

__inline__ int ddot (const int n, const float * const x, const float * const y, double * const result) {
  struct argStruct args;
  partDDOT = 0;
  args.arrX = x;
  args.arrY = y;
  MAXDDOT = n;
  int i;

  for(int i = 0; i < MAX_THREAD; i++) {
    sumDDOTArray[i] = 0.0;
  }

  if (pthread_mutex_init(&lockDDOT, NULL) != 0) {
    return 1;
  }
  
  pthread_t threads[MAX_THREAD];
  double local_result = 0.0;

  if (__builtin_expect(y==x, 0)){
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, sumXXArray, (void*)x);
      }

      for (i = 0; i < MAX_THREAD; i++) {
        pthread_join(threads[i], NULL);
      }

      for (i = 0; i < MAX_THREAD; i++) {
        local_result += sumDDOTArray[i];
      }

  } else {
      for (i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, sumXYArray, (void*)&args);
      }

      for (i = 0; i < MAX_THREAD; i++) {
        pthread_join(threads[i], NULL);
      }

      for (i = 0; i < MAX_THREAD; i++) {
        local_result += sumDDOTArray[i];
      }
  }
  if ((MAXDDOT % MAX_THREAD) != 0) {
    register __m256 xVecSquareSum1, 
                    xVecSquareSum2, 
                    xVecSquare,     
                    xVec1, 
                    xVec2, 
                    xVec3, 
                    xVec4, 
                    xVec5, 
                    xVec6, 
                    xVec7, 
                    xVec8;
    register __m128 hiQuadSum,      
                    hiDualSum,      
                    singleSum;
    if (__builtin_expect(y==x, 0)){
        for (i=(MAXDDOT/MAX_THREAD)*MAX_THREAD ; i<MAXDDOT; i+=loopFactorDDOT) {
          xVec1  = _mm256_loadu_ps(x + i     );
          xVec2  = _mm256_loadu_ps(x + i +  8);
          xVec3  = _mm256_loadu_ps(x + i + 16);
          xVec4  = _mm256_loadu_ps(x + i + 24);
          xVec5  = _mm256_loadu_ps(x + i + 32);
          xVec6  = _mm256_loadu_ps(x + i + 40);
          xVec7  = _mm256_loadu_ps(x + i + 48);
          xVec8  = _mm256_loadu_ps(x + i + 56);

          xVecSquareSum1 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec1 , xVec1 ),_mm256_mul_ps(xVec2 , xVec2 )),_mm256_mul_ps(xVec3 , xVec3 )),_mm256_mul_ps(xVec4 , xVec4 ));
          xVecSquareSum2 = _mm256_add_ps( _mm256_add_ps( _mm256_add_ps( _mm256_mul_ps(xVec5 , xVec5 ),_mm256_mul_ps(xVec6 , xVec6 )),_mm256_mul_ps(xVec7 , xVec7 )),_mm256_mul_ps(xVec8 , xVec8 ));
          xVecSquare     = _mm256_add_ps(xVecSquareSum1, xVecSquareSum2);
          hiQuadSum      = _mm_add_ps(_mm256_castps256_ps128(xVecSquare), _mm256_extractf128_ps(xVecSquare, 1));
          hiDualSum      = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
          singleSum      = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));

            local_result = local_result + _mm_cvtss_f32(singleSum);
        }

        for (; i<MAXDDOT; i++) {
          local_result = local_result + x[i]*x[i];
        }

    } else {
        for (i=(MAXDDOT/MAX_THREAD)*MAX_THREAD; i<MAXDDOT; i+=loopFactorDDOT) {
          xVecSquareSum1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i     ),_mm256_loadu_ps(y + i     )),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 8  ), _mm256_loadu_ps(y + i + 8  ))),_mm256_mul_ps(_mm256_loadu_ps(x + i+ 16  ), _mm256_loadu_ps(y + i+ 16  ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 24 ), _mm256_loadu_ps(y + i + 24 )));
          xVecSquareSum2 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 32),_mm256_loadu_ps(y + i + 32)),_mm256_mul_ps(  _mm256_loadu_ps(x + i + 40 ), _mm256_loadu_ps(y + i + 40 ))),_mm256_mul_ps(_mm256_loadu_ps(x + i + 48 ), _mm256_loadu_ps(y + i + 48 ))), _mm256_mul_ps(  _mm256_loadu_ps(x + i + 56 ), _mm256_loadu_ps(y + i + 56 )));
          xVecSquare = _mm256_add_ps(xVecSquareSum1, xVecSquareSum2);
          hiQuadSum = _mm_add_ps(_mm256_castps256_ps128(xVecSquare), _mm256_extractf128_ps(xVecSquare, 1));
          hiDualSum = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
          singleSum = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
          local_result = local_result + _mm_cvtss_f32(singleSum);
        }
        for (; i<MAXDDOT; i++) {
          local_result = local_result + x[i]*y[i];
        }
    }
  }
  
  *result = local_result;
          //printf("this is value: %0.6f\n", local_result);
  for(int i = 0; i < MAX_THREAD; i++) {
    sumDDOTArray[i] = 0.0;
  }

  pthread_mutex_destroy(&lockDDOT);
  partDDOT = 0;
  return 0;
}