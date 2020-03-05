/*
 *  warca is a library for metric learning using weighted approximate rank
 *  component analysis algorithm written in c++.
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijose@idiap.ch>
 *
 *  This file is part of warca.
 *
 *  warca is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  warca is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "math_functions.h"
#include <algorithm>

template <>
void cblas_gemm<float>(const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const float alpha, const float *A,
                       const float *B, const float beta, float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void cblas_gemm<double>(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha, const double *A,
                        const double *B, const double beta, double *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void cblas_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                       const float alpha, const float *A, const float *x,
                       const float beta, float *y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void cblas_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const double alpha, const double *A, const double *x,
                        const double beta, double *y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void cblas_axpy<float>(const int N, const float alpha, const float *X,
                       float *Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void cblas_axpy<double>(const int N, const double alpha, const double *X,
                        double *Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <>
void cblas_axpby<float>(const int N, const float alpha, const float *X,
                        const float beta, float *Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void cblas_axpby<double>(const int N, const double alpha, const double *X,
                         const double beta, double *Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <> void cblas_copy<float>(const int N, const float *X, float *Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template <> void cblas_copy<double>(const int N, const double *X, double *Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}

template <> void cblas_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void cblas_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
float cblas_dot<float>(const int n, const float *x, const float *y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template <>
double cblas_dot<double>(const int n, const double *x, const double *y) {
  return cblas_ddot(n, x, 1, y, 1);
}

template <> void cblas_exp<double>(const int N, const double *X, double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = exp(X[i]);
}

template <> void cblas_exp<float>(const int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = exp(X[i]);
}

template <> void cblas_diag<double>(const int N, const double *X, double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = X[i + i * N];
}

template <> void cblas_diag<float>(const int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = X[i + i * N];
}

template <> void cblas_fill<float>(const int N, const float alpha, float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = alpha;
}

template <>
void cblas_fill<double>(const int N, const double alpha, double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = alpha;
}

template <> void cblas_log<double>(const int N, const double *X, double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = log(X[i]);
}

template <> void cblas_log<float>(const int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = log(X[i]);
}

template <> void cblas_sqr<double>(const int N, const double *X, double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = sqrt(X[i]);
}

template <> void cblas_sqr<float>(const int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = sqrt(X[i]);
}

template <>
void cblas_pow<double>(const int N, const double *X, const double alpha,
                       double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = pow(X[i], alpha);
}

template <>
void cblas_pow<float>(const int N, const float *X, const float alpha,
                      float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = pow(X[i], alpha);
}

template <>
void cblas_add_scalar(const int N, const float *X, const float alpha,
                      float *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = X[i] + alpha;
}

template <>
void cblas_add_scalar(const int N, const double *X, const double alpha,
                      double *Y) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Y[i] = X[i] + alpha;
}

template <>
void cblas_add<double>(const int N, const double *X, const double *Y,
                       double *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] + Y[i];
}

template <>
void cblas_add<float>(const int N, const float *X, const float *Y, float *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] + Y[i];
}

template <typename scalar_t>
void cblas_sub(const int N, const scalar_t *X, const scalar_t *Y, scalar_t *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] - Y[i];
}

template <typename scalar_t>
void cblas_mul(const int N, const scalar_t *X, const scalar_t *Y, scalar_t *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] * Y[i];
}

template <>
void cblas_div<double>(const int N, const double *X, const double *Y,
                       double *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] / Y[i];
}

template <>
void cblas_div<float>(const int N, const float *X, const float *Y, float *Z) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for (int i = 0; i < N; ++i)
    Z[i] = X[i] / Y[i];
}

inline float sse2_smax(const int N, const float *a) {
  float max_val = a[0];
  for (int i = 1; i < N; i++)
    max_val = std::max(a[i], max_val);
  return max_val;
}

inline double sse2_dmax(const int N, const double *a) {
  double max_val = a[0];
  for (int i = 1; i < N; i++)
    max_val = std::max(a[i], max_val);
  return max_val;
}

template <> float cblas_max<float>(const int N, const float *X) {
  return sse2_smax(N, X);
}

template <> double cblas_max<double>(const int N, const double *X) {
  return sse2_dmax(N, X);
}
