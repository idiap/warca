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

#ifndef _MATH_FUNCTIONS_H_
#define _MATH_FUNCTIONS_H_

extern "C" {
#include <cblas.h>
}
#include <math.h>
#include <omp.h>
#include <xmmintrin.h>

template <typename scalar_t>
void cblas_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                const int M, const int N, const int K, const scalar_t alpha,
                const scalar_t *A, const scalar_t *B, const scalar_t beta,
                scalar_t *C);

template <typename scalar_t>
void cblas_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                const scalar_t alpha, const scalar_t *A, const scalar_t *x,
                const scalar_t beta, scalar_t *y);

template <typename scalar_t>
void cblas_axpy(const int N, const scalar_t alpha, const scalar_t *X,
                scalar_t *Y);

template <typename scalar_t>
void cblas_axpby(const int N, const scalar_t alpha, const scalar_t *X,
                 const scalar_t beta, scalar_t *Y);

template <typename scalar_t>
void cblas_copy(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_scal(const int N, const scalar_t alpha, scalar_t *X);

template <typename scalar_t> scalar_t cblas_max(const int N, const scalar_t *X);

template <typename scalar_t>
void cblas_exp(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_diag(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_fill(const int N, const scalar_t alpha, scalar_t *Y);

template <typename scalar_t>
void cblas_log(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_sqr(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_abs(const int N, const scalar_t *X, scalar_t *Y);

template <typename scalar_t>
void cblas_pow(const int N, const scalar_t *X, const scalar_t p, scalar_t *Y);

template <typename scalar_t>
scalar_t cblas_dot(const int n, const scalar_t *x, const scalar_t *y);

template <typename scalar_t>
void cblas_add(const int N, const scalar_t *a, const scalar_t *b, scalar_t *y);

template <typename scalar_t>
void cblas_div(const int N, const scalar_t *a, const scalar_t *b, scalar_t *y);

template <typename scalar_t>
void cblas_add_scalar(const int N, const scalar_t *X, const scalar_t alpha,
                      scalar_t *Y);

inline void cblas_saxpby(const int N, const float alpha, const float *X,
                         const int incX, const float beta, float *Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double *X,
                         const int incX, const double beta, double *Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif
