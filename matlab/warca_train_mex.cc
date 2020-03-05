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

#include "assert.h"
#include "mex.h"
#include "warca.h"
#include <cstring>

using namespace warca;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  const mxArray *mxKernel = prhs[0];
  const mxArray *mxLabels = prhs[1];
  const mxArray *mxRank = prhs[2];
  const mxArray *mxLambda = prhs[3];
  const mxArray *mxEta = prhs[4];
  const mxArray *mxIter = prhs[5];
  const mxArray *mxBatchSize = prhs[6];
  const mxArray *mxMaxSampling = prhs[7];
  const mxArray *mxKernelType = prhs[8];
  const mxArray *mxSeed = prhs[9];
  const mxArray *mxQuiet = prhs[10];

  const double *K = (const double *)mxGetPr(mxKernel);
  const uint32_t *y = (const uint32_t *)mxGetPr(mxLabels);
  const uint32_t r = (const uint32_t)mxGetScalar(mxRank);
  const double lambda = (const double)mxGetScalar(mxLambda);
  const double eta = (const double)mxGetScalar(mxEta);
  const uint32_t max_iter = (const uint32_t)mxGetScalar(mxIter);
  const uint32_t batch_size = (const uint32_t)mxGetScalar(mxBatchSize);
  const uint32_t max_sampling = (const uint32_t)mxGetScalar(mxMaxSampling);
  KernelType kernel_type = KernelType((uint32_t)mxGetScalar(mxKernelType));
  const double seed = (const double)mxGetScalar(mxSeed);
  const bool quiet = (const bool)mxGetScalar(mxQuiet);

  const uint32_t m = (const int)mxGetM(mxKernel);
  const uint32_t l = (const int)mxGetN(mxKernel);
  const uint32_t n = (const int)mxGetM(mxLabels);
  assert(m == l);
  assert(l == n);
  if (kernel_type != PRE_COMP && kernel_type != LIN) {
    throw warca_error("The matlab interface currently supports only "
                      "precomputed kernels of linear warca\n");
  }
  WARCA<double> warca_object(seed, quiet);
  warca_object.train(K, y, n, n, r, lambda, eta, max_iter, batch_size,
                     max_sampling, kernel_type, 0, 3, 0);
  const double *w = warca_object.weight();

  plhs[0] = mxCreateNumericMatrix(n, r, mxDOUBLE_CLASS, mxREAL);
  double *A = (double *)mxGetPr(plhs[0]);
  memcpy(A, w, sizeof(double) * r * n);
}
