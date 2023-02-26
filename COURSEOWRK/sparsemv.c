#undef _GLIBCXX_DEBUG  

#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <pthread.h>

#pragma GCC optimize("O3,inline")
#pragma GCC target("bmi,bmi2,lzcnt,popcnt")

#include "sparsemv.h"

#define MAX_THREAD 4

#define LOOPFACTOR4  4
#define LOOPFACTOR8  8
#define LOOPFACTOR16 16
#define LOOPFACTOR27 27

int MAXSPARSE;
int partSparse = 0;
pthread_mutex_t lockSparse;

struct argStruct {
  const float *arrayX;
  const float *arrayY;
  struct mesh *matrix;
};

/**
 * @brief Compute matrix vector product (y = A*x). Now with parallelisation and SIMD vector intrinsics!
 *
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */

void* sparseProcess(void* arg) {
  pthread_mutex_lock(&lockSparse);
  register int threadPart = partSparse++;
  pthread_mutex_unlock(&lockSparse);
  register int i = 0;
  register struct argStruct *args = arg;
  struct mesh *A = (struct mesh *)(args->matrix);
  register float *x = (float *)(args->arrayX);
  register float *y = (float *)(args->arrayY);
  register int start = threadPart * (MAXSPARSE/MAX_THREAD), end =(threadPart + 1) * (MAXSPARSE / MAX_THREAD);
  register int j, loopN4, loopN8, loopN16, loopN27, cur_nnz;
  register int * cur_inds;
  register float * cur_vals;
  register float sum;
  register __m256i indexes1,
                   indexes2,
                   indexes3;
  register __m256 trout1,
                  trout2,     
                  trout3, 
                  tuna1,      
                  tuna2,      
                  tuna3,      
                  tunaTrout1, 
                  tunaTrout2, 
                  tunaTrout3, 
                  threeSum;
  register __m128i indexes128;
  register __m128 hiQuadSum,
                  hiDualSum,
                  singleSum,
                  trout128,
                  tuna128,
                  troutTuna128;

  for (i= start; i<end; i++) {
      sum = 0.0;
      j = 0;
      cur_vals = (float *) A->ptr_to_vals_in_row[i];
      cur_inds = (int *) A->ptr_to_inds_in_row[i];
      cur_nnz  = (int) A->nnz_in_row[i];

      loopN4  = (cur_nnz >> 2) << 2;
      loopN8  = (cur_nnz >> 3) << 3;
      loopN16 = (cur_nnz >> 4) << 4;
      loopN27 = (cur_nnz/27) * 27;
      //register int loopN = cur_nnz/LOOPFACTOR;
      for (; j < loopN27; j+= LOOPFACTOR27)
      {
        
        indexes1  = _mm256_loadu_si256((const __m256i*)(cur_inds + j));
        indexes2  = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 8));
        indexes3  = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 16));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        trout2     = _mm256_i32gather_ps(x, indexes2, 4);
        trout3     = _mm256_i32gather_ps(x, indexes3, 4);
        tuna1      = _mm256_loadu_ps(cur_vals);
        tuna2      = _mm256_loadu_ps(cur_vals + j + 8);
        tuna3      = _mm256_loadu_ps(cur_vals + j + 16);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        tunaTrout2 = _mm256_mul_ps(tuna2, trout2);
        tunaTrout3 = _mm256_mul_ps(tuna3, trout3);
        threeSum   = _mm256_add_ps(_mm256_add_ps(tunaTrout1, tunaTrout3), tunaTrout2);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(threeSum), _mm256_extractf128_ps(threeSum, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum)
                  + cur_vals[24]*x[cur_inds[24]]
                  + cur_vals[25]*x[cur_inds[25]]
                  + cur_vals[26]*x[cur_inds[26]];
      }
      for (; j < loopN16; j+= LOOPFACTOR16)
      {
        indexes1   = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 0));
        indexes2   = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 8));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        trout2     = _mm256_i32gather_ps(x, indexes2, 4);
        tuna1      = _mm256_loadu_ps(cur_vals + j);
        tuna2      = _mm256_loadu_ps(cur_vals + j + 8);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        tunaTrout2 = _mm256_mul_ps(tuna2, trout2);
        threeSum   = _mm256_add_ps(tunaTrout1, tunaTrout2);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(threeSum), _mm256_extractf128_ps(threeSum, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum +  _mm_cvtss_f32(singleSum);
      }
      for (; j < loopN8; j+= LOOPFACTOR8)
      {
        indexes1  = _mm256_loadu_si256((const __m256i*)(cur_inds + j));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        tuna1      = _mm256_loadu_ps(cur_vals + j);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(tunaTrout1), _mm256_extractf128_ps(tunaTrout1, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum);
      }
      for (; j < loopN4; j+= LOOPFACTOR4)
      {
        indexes128    = _mm_loadu_si128((const __m128i*)(cur_inds + j));
        trout128      = _mm_i32gather_ps(x, indexes128, 4);
        tuna128       = _mm_loadu_ps(cur_vals + j);
        troutTuna128  = _mm_mul_ps(tuna128, trout128);
        hiDualSum     = _mm_add_ps(troutTuna128, _mm_movehl_ps(troutTuna128, troutTuna128));
        singleSum     = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum);
      }
      for (; j < cur_nnz; j++)
      {
        sum = sum + cur_vals[j]*x[cur_inds[j]];
      }
      y[i] = sum;
    }
  return 0;
}

__inline__ int sparsemv(struct mesh *A, const float * const x, float * const y)
{
  MAXSPARSE = (const int) A->local_nrow;
  struct argStruct args;
  args.arrayX = x;
  args.arrayY = y;
  args.matrix = A;
  int i;

  if (pthread_mutex_init(&lockSparse, NULL) != 0) {
    return 1;
  }
  pthread_t threads[MAX_THREAD];

  for (i = 0; i < MAX_THREAD; i++) {
    pthread_create(&threads[i], NULL, sparseProcess, (void*)&args);
  }
  for (i = 0; i < MAX_THREAD; i++) {
    pthread_join(threads[i], NULL);
  }
  if ((MAXSPARSE % MAX_THREAD) != 0) {
    register __m256i indexes1,
                   indexes2,
                   indexes3;
    register __m256 trout1,
                    trout2,     
                    trout3, 
                    tuna1,      
                    tuna2,      
                    tuna3,      
                    tunaTrout1, 
                    tunaTrout2, 
                    tunaTrout3, 
                    threeSum;
    register __m128i indexes128;
    register __m128 hiQuadSum,
                    hiDualSum,
                    singleSum,
                    trout128,
                    tuna128,
                    troutTuna128;
      for (i= (MAXSPARSE/MAX_THREAD)*MAX_THREAD; i<MAXSPARSE; i++) {
      register float sum = 0.0;
      register float * cur_vals = (float *) A->ptr_to_vals_in_row[i];
      register int * cur_inds = (int *) A->ptr_to_inds_in_row[i];
      register int cur_nnz = (int) A->nnz_in_row[i];
      register int j, loopN4, loopN8, loopN16, loopN27;
      j = 0;
      loopN4  = (cur_nnz >> 2) << 2;
      loopN8  = (cur_nnz >> 3) << 3;
      loopN16 = (cur_nnz >> 4) << 4;
      loopN27 = (cur_nnz/27) * 27;
      //register int loopN = cur_nnz/LOOPFACTOR;
      for (; j < loopN27; j+= LOOPFACTOR27)
      { 
        indexes1  = _mm256_loadu_si256((const __m256i*)(cur_inds + j));
        indexes2  = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 8));
        indexes3  = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 16));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        trout2     = _mm256_i32gather_ps(x, indexes2, 4);
        trout3     = _mm256_i32gather_ps(x, indexes3, 4);
        tuna1      = _mm256_loadu_ps(cur_vals);
        tuna2      = _mm256_loadu_ps(cur_vals + j + 8);
        tuna3      = _mm256_loadu_ps(cur_vals + j + 16);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        tunaTrout2 = _mm256_mul_ps(tuna2, trout2);
        tunaTrout3 = _mm256_mul_ps(tuna3, trout3);
        threeSum   = _mm256_add_ps(_mm256_add_ps(tunaTrout1, tunaTrout3), tunaTrout2);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(threeSum), _mm256_extractf128_ps(threeSum, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum)
                  + cur_vals[24]*x[cur_inds[24]]
                  + cur_vals[25]*x[cur_inds[25]]
                  + cur_vals[26]*x[cur_inds[26]];
      }
      for (; j < loopN16; j+= LOOPFACTOR16)
      {
        indexes1   = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 0));
        indexes2   = _mm256_loadu_si256((const __m256i*)(cur_inds + j + 8));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        trout2     = _mm256_i32gather_ps(x, indexes2, 4);
        tuna1      = _mm256_loadu_ps(cur_vals + j);
        tuna2      = _mm256_loadu_ps(cur_vals + j + 8);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        tunaTrout2 = _mm256_mul_ps(tuna2, trout2);
        threeSum   = _mm256_add_ps(tunaTrout1, tunaTrout2);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(threeSum), _mm256_extractf128_ps(threeSum, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum +  _mm_cvtss_f32(singleSum);
      }
      for (; j < loopN8; j+= LOOPFACTOR8)
      {
        indexes1  = _mm256_loadu_si256((const __m256i*)(cur_inds + j));
        trout1     = _mm256_i32gather_ps(x, indexes1, 4);
        tuna1      = _mm256_loadu_ps(cur_vals + j);
        tunaTrout1 = _mm256_mul_ps(tuna1, trout1);
        hiQuadSum  = _mm_add_ps(_mm256_castps256_ps128(tunaTrout1), _mm256_extractf128_ps(tunaTrout1, 1));
        hiDualSum  = _mm_add_ps(hiQuadSum, _mm_movehl_ps(hiQuadSum, hiQuadSum));
        singleSum  = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum);
      }
      for (; j < loopN4; j+= LOOPFACTOR4)
      {
        indexes128    = _mm_loadu_si128((const __m128i*)(cur_inds + j));
        trout128      = _mm_i32gather_ps(x, indexes128, 4);
        tuna128       = _mm_loadu_ps(cur_vals + j);
        troutTuna128  = _mm_mul_ps(tuna128, trout128);
        hiDualSum     = _mm_add_ps(troutTuna128, _mm_movehl_ps(troutTuna128, troutTuna128));
        singleSum     = _mm_add_ps(hiDualSum, _mm_shuffle_ps(hiDualSum, hiDualSum, 0x1));
        sum = sum + _mm_cvtss_f32(singleSum);
      }
      for (; j < cur_nnz; j++)
      {
        sum = sum + cur_vals[j]*x[cur_inds[j]];
      }
      y[i] = sum;
    }
  }
  partSparse = 0;
  MAXSPARSE = 0;
  pthread_mutex_destroy(&lockSparse);
  return 0;
}