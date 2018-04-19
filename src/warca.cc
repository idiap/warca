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

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string>
#include <assert.h>
#include <limits>
#include "warca.h"
#include "math_functions.h"


namespace warca {

template<typename scalar_t>
WARCA<scalar_t>::WARCA(scalar_t seed, bool quiet) {
  dim_ = 0;
  num_svs_ = 0;
  dim_svs_ = 0;
  rank_ = 0;
  gamma_ = scalar_t(0);
  degree_ = 3;
  coeff_ = scalar_t(0);
  weight_ = NULL;
  SVs_ = NULL;
  rnd_ = new Prand(seed);
  quiet_ = quiet;
}

template<typename scalar_t>
WARCA<scalar_t>::WARCA(scalar_t seed) {
  dim_ = 0;
  num_svs_ = 0;
  dim_svs_ = 0;
  rank_ = 0;
  gamma_ = scalar_t(0);
  degree_ = 3;
  coeff_ = scalar_t(0);
  weight_ = NULL;
  SVs_ = NULL;
  rnd_ = new Prand(seed);
  quiet_ = true;
}

template<typename scalar_t>
WARCA<scalar_t>::WARCA() {
  dim_ = 0;
  num_svs_ = 0;
  dim_svs_ = 0;
  rank_ = 0;
  gamma_ = scalar_t(0);
  degree_ = 3;
  coeff_ = scalar_t(0);
  weight_ = NULL;
  SVs_ = NULL;
  rnd_ = new Prand(scalar_t(1));
  quiet_ = true;

}

template<typename scalar_t>
WARCA<scalar_t>::WARCA(bool quiet) {
  dim_ = 0;
  num_svs_ = 0;
  dim_svs_ = 0;
  rank_ = 0;
  gamma_ = scalar_t(1);
  degree_ = 3;
  coeff_ = scalar_t(0);
  weight_ = NULL;
  SVs_ = NULL;
  rnd_ = new Prand(scalar_t(1));
  quiet_ = quiet;
}

template <typename scalar_t>
void WARCA<scalar_t>::load_model(const char *filename) {
  std::cout<<"Loading model\n";
  if(weight_ != NULL) {  delete [] weight_; }
  if(SVs_ != NULL) { delete [] SVs_;}
  std::ifstream fin(filename);
  uint32_t kernel_type;
  fin>>dim_;
  fin>>rank_;
  fin>>num_svs_;
  fin>>dim_svs_;
  fin>>kernel_type;
  fin>>gamma_;
  fin>>degree_;
  fin>>coeff_;
  kernel_type_ = KernelType(kernel_type);
  weight_ =  new scalar_t [rank_ * dim_];
  SVs_ = NULL;
  if(kernel_type_ != PRE_COMP && kernel_type_ != LIN ) {
    if(num_svs_ == 0 || dim_svs_ == 0) { 
      throw warca_error("Number and dimension of support vectors should be greater than zero\n");
    }
    SVs_ = new scalar_t [num_svs_ * dim_svs_];
  }
  uint32_t n = 0;
  while(fin) {
    scalar_t data;
    std::string line;
    if(!std::getline(fin, line, ',')) break;
    std::istringstream ss(line);
    while(ss >> data ){
      if(n < rank_ * dim_)
        weight_[n] = data;
      else if(n < rank_ * dim_ + num_svs_ * dim_svs_)
        SVs_[n - rank_ * dim_] = data;
      n++;
    }
  }
  fin.close();
}
template<typename scalar_t>
WARCA<scalar_t>::WARCA(const char* filename, scalar_t seed) {
  weight_ = NULL;
  SVs_ = NULL;
  load_model(filename);
  rnd_ = new Prand(seed);
}

template<typename scalar_t>
WARCA<scalar_t>::WARCA(const char* filename) {
  weight_ = NULL;
  SVs_ = NULL;
  load_model(filename);
  rnd_ = new Prand(1);
}

template<typename scalar_t>
void WARCA<scalar_t>::save_model(const char* filename) {
  std::ofstream fout(filename);
  fout<<dim_<<std::endl;
  fout<<rank_<<std::endl;
  fout<<num_svs_<<std::endl;
  fout<<dim_svs_<<std::endl;
  fout<<kernel_type_<<std::endl;
  fout<<gamma_<<std::endl;
  fout<<degree_<<std::endl;
  fout<<coeff_<<std::endl;
  if(weight_ == NULL) {
      throw warca_error("Weight matrix should not be NULL\n");
   }
  for(uint32_t i = 0; i < rank_ * dim_ - 1; i++) { fout<<weight_[i]<<","; }
  fout<<weight_[rank_ * dim_ - 1]<<std::endl;
  if(kernel_type_ != PRE_COMP && kernel_type_ != LIN ) {
    if(SVs_ == NULL) {
      throw warca_error("SVs should not be NULL for kernel type other than PRE_COMP and LIN\n");
    }
    for(uint32_t i = 0; i < num_svs_ * dim_svs_ - 1; i++) { fout<<SVs_[i]<<","; }
    fout<<SVs_[num_svs_ * dim_svs_ - 1]<<std::endl;
  }
  fout.close();
}

template<typename scalar_t>
scalar_t WARCA<scalar_t>::l2dist(const scalar_t *x, const scalar_t *y, uint32_t d) {
  scalar_t xx = cblas_dot<scalar_t>(d, x, x);
  scalar_t yy = cblas_dot<scalar_t>(d, y, y);
  scalar_t xy = cblas_dot<scalar_t>(d, x, y);
  scalar_t dist = xx + yy - scalar_t(2) * xy;
  return (scalar_t)sqrt(dist);
}

template<typename scalar_t>
void WARCA<scalar_t>::compute_chi2rbf_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,
                                          uint32_t num_x, uint32_t num_y, uint32_t dim) {
  if(gamma_ == 0) { gamma_ = scalar_t(1) / scalar_t(dim); }
   for(uint32_t i = 0; i < num_x; i++) {
    const scalar_t *x_matrix_i = x_matrix + i * dim;
    scalar_t *kernel_matrix_i = kernel_matrix + i  * num_y;
    for(uint32_t j = 0; j < num_y; j++) {
      const scalar_t *y_matrix_j = y_matrix + j * dim;
      kernel_matrix_i[j] = scalar_t(0);
      for(uint32_t k = 0; k < dim; k++) {
        scalar_t num = x_matrix_i[k]  - y_matrix_j[k];
        scalar_t den = x_matrix_i[k]  + y_matrix_j[k];
        if(den != scalar_t(0)) { kernel_matrix_i[j] += num * num / den; }
      }
      kernel_matrix_i[j] = exp(-kernel_matrix_i[j]  * gamma_);
    }
  }
}

template<typename scalar_t>
void WARCA<scalar_t>::compute_chi2rbf_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix,
                                          uint32_t num_x, uint32_t dim) {
  if(gamma_ == 0) { gamma_ = scalar_t(1) / scalar_t(dim); }
  cblas_scal<scalar_t>(num_x * num_x, scalar_t(0), kernel_matrix);
  for(uint32_t i = 0; i < num_x; i++) {
    const scalar_t *x_matrix_i = x_matrix + i * dim;
    scalar_t *kernel_matrix_i = kernel_matrix + i  * num_x;
    kernel_matrix_i[i] = scalar_t(1);
    for(uint32_t j = i + 1; j < num_x; j++) {
      const scalar_t *x_matrix_j = x_matrix + j * dim;
      kernel_matrix_i[j] = scalar_t(0);
      for(uint32_t k = 0; k < dim; k++) {
        scalar_t num = x_matrix_i[k]  - x_matrix_j[k];
        scalar_t den = x_matrix_i[k]  + x_matrix_j[k];
        if(den != scalar_t(0)) { kernel_matrix_i[j] += num * num / den; }
      }
      scalar_t *kernel_matrix_j = kernel_matrix + j  * num_x;
      kernel_matrix_i[j] = exp(-kernel_matrix_i[j] * gamma_);
      kernel_matrix_j[i] = kernel_matrix_i[j];
    }
  }
}


template<typename scalar_t>
void WARCA<scalar_t>::compute_polynomial_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,
                                             uint32_t num_x, uint32_t num_y, uint32_t dim) {
  cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_x, num_y, dim, scalar_t(gamma_), x_matrix, y_matrix, scalar_t(0), kernel_matrix);
  cblas_add_scalar<scalar_t>(num_x * num_y, kernel_matrix, coeff_, kernel_matrix);
  cblas_pow<scalar_t>(num_x * num_y, kernel_matrix, scalar_t(degree_), kernel_matrix);
  
}

template<typename scalar_t>
void WARCA<scalar_t>::compute_polynomial_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix,
                                             uint32_t num_x, uint32_t dim) {
  cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_x, num_x, dim, scalar_t(gamma_), x_matrix, x_matrix, scalar_t(0), kernel_matrix);
  cblas_add_scalar<scalar_t>(num_x * num_x, kernel_matrix, coeff_, kernel_matrix);
  cblas_pow<scalar_t>(num_x * num_x, kernel_matrix, scalar_t(degree_), kernel_matrix);
}


template<typename scalar_t>
void WARCA<scalar_t>::compute_rbf_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,
                                      uint32_t num_x, uint32_t num_y, uint32_t dim) {
  if(gamma_ == 0) { gamma_ = scalar_t(1) / scalar_t(dim); }
  for(uint32_t i = 0; i < num_x; i++) {
    const scalar_t *x_matrix_i = x_matrix + i * dim;
    scalar_t *kernel_matrix_i = kernel_matrix + i  * num_y;
    for(uint32_t j = 0; j < num_y; j++) {
      const scalar_t *y_matrix_j = y_matrix + j * dim;
      scalar_t dist_ij = l2dist(x_matrix_i, y_matrix_j, dim);
      kernel_matrix_i[j] = exp(-dist_ij * dist_ij  * gamma_);
    }
  }
}

template<typename scalar_t>
void WARCA<scalar_t>::compute_rbf_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix,
                                      uint32_t num_x, uint32_t dim) {
  if(gamma_ == 0) { gamma_ = scalar_t(1) / scalar_t(dim); }
  cblas_scal<scalar_t>(num_x * num_x, scalar_t(0), kernel_matrix);
  for(uint32_t i = 0; i < num_x; i++) {
    const scalar_t *x_matrix_i = x_matrix + i * dim;
    scalar_t *kernel_matrix_i = kernel_matrix + i  * num_x;
    kernel_matrix_i[i] = scalar_t(1);
    for(uint32_t j = i + 1; j < num_x; j++) {
      const scalar_t *x_matrix_j = x_matrix + j * dim;
      scalar_t dist_ij = l2dist(x_matrix_i, x_matrix_j, dim);
      kernel_matrix_i[j] = exp(-dist_ij * dist_ij  * gamma_);
      scalar_t *kernel_matrix_j = kernel_matrix + j  * num_x;
      kernel_matrix_j[i] = kernel_matrix_i[j];
    }
  }
}

template<typename scalar_t>
scalar_t WARCA<scalar_t>::warp_gradient_dual(const scalar_t *M, const uint32_t *labels, scalar_t *G,
                                       uint32_t batch_size, const uint32_t max_sampling, SimilarPairs* pairs){
  scalar_t loss = scalar_t(0);
  uint32_t i, j;
  cblas_scal<scalar_t>(dim_ * dim_, scalar_t(0), G);
  for(uint32_t l = 0; l < batch_size; l++) {
    pairs->sample_pair(&i, &j, rnd_);
    const scalar_t *m_i = M + i * rank_;
    const scalar_t *m_j = M + j * rank_;
    scalar_t *g_i = G + i * dim_;
    scalar_t *g_j = G + j * dim_;
    const scalar_t sqrt_ij = l2dist(m_i, m_j, rank_) + scalar_t(1e-9);
    scalar_t sqrt_ik = scalar_t(0), sqrt_jk = scalar_t(0), loss_ij = scalar_t(0), loss_ji = scalar_t(0);
    uint32_t times_sampled = 0, k;
    bool flag = false;
    do {
      k = rnd_->randi(0, dim_);
      while(labels[i] == labels[k]) { k = rnd_->randi(0, dim_); }
      times_sampled++;
      const scalar_t *m_k = M + k * rank_;
      sqrt_ik = l2dist(m_i, m_k, rank_) + scalar_t(1e-9);
      sqrt_jk = l2dist(m_j, m_k, rank_) + scalar_t(1e-9);
      loss_ij = std::max(scalar_t(0), scalar_t(1) + sqrt_ij - sqrt_ik);
      loss_ji = std::max(scalar_t(0), scalar_t(1) + sqrt_ij - sqrt_jk);
      if(loss_ij > scalar_t(0) || loss_ji > scalar_t(0)) {
        loss += loss_ij + loss_ji;
        scalar_t warp_coeff = scalar_t(1) +  scalar_t(log(scalar_t(max_sampling)/scalar_t(times_sampled)));
        if(loss_ij > scalar_t(0)) {loss_ij = scalar_t(1);}
        if(loss_ji > scalar_t(0)) {loss_ji = scalar_t(1);}
        scalar_t *g_k = G + dim_ * k;
        g_i[i] +=  warp_coeff * ((loss_ij + loss_ji)/ sqrt_ij - loss_ij / sqrt_ik);
        g_i[j] -=  warp_coeff * ((loss_ij + loss_ji) / sqrt_ij);
        g_i[k] +=  warp_coeff * (loss_ij / sqrt_ik);
        g_j[j] +=  warp_coeff * ((loss_ji + loss_ij) / sqrt_ij - loss_ji / sqrt_jk); 
        g_j[i] -=  warp_coeff * ((loss_ij + loss_ji) / sqrt_ij);
        g_j[k] +=  warp_coeff * (loss_ji / sqrt_jk);
        g_k[k] -=  warp_coeff * (loss_ij / sqrt_ik + loss_ji / sqrt_jk);
        g_k[i] +=  warp_coeff * loss_ij / sqrt_ik;
        g_k[j] +=  warp_coeff * loss_ji / sqrt_jk;
        flag = true;
      }
    } while((times_sampled < max_sampling) && !flag);
  }
  return loss;
}


template<typename scalar_t>
void WARCA<scalar_t>::train_adam_dual(const scalar_t *kernel_matrix, const uint32_t *labels,  scalar_t lambda,
                                   scalar_t eta,  uint32_t max_iter, uint32_t batch_size, uint32_t max_sampling) {
  SimilarPairs *pairs = new SimilarPairs(labels, dim_);
  scalar_t beta1 = scalar_t(0.9), beta2 = scalar_t(0.999), eps = scalar_t(1e-8);
  uint32_t rtd = rank_ * dim_;
  scalar_t *M_adam =  new scalar_t [4 * rtd + dim_ * dim_ + rank_ * rank_];
  scalar_t *V_adam =  M_adam  + rtd;
  scalar_t *G_sqr =  V_adam + rtd;
  scalar_t *M = G_sqr + rtd;
  scalar_t *G = M + rtd;
  scalar_t *R = G + dim_ * dim_;
  scalar_t loss = std::numeric_limits<scalar_t>::infinity();
  scalar_t stddev = sqrt(scalar_t(1.) / scalar_t(dim_));
  max_sampling = std::min(max_sampling, dim_);
  for(uint32_t i = 0; i < rtd; i++) {
    weight_[i] = scalar_t(rnd_->randn(0, stddev));
  }
  cblas_fill<scalar_t>(2 * rtd, scalar_t(0), M_adam);
  uint32_t iter = 0;
  while(iter < max_iter) {
    cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, dim_, rank_, dim_, scalar_t(1), kernel_matrix, weight_, scalar_t(0), M);
    loss = warp_gradient_dual(M, labels, G,  batch_size, max_sampling, pairs);
    cblas_gemm<scalar_t>(CblasTrans, CblasNoTrans, rank_, dim_, dim_,  scalar_t(2.0 / batch_size), M, G, scalar_t(0), G_sqr);
    cblas_gemm<scalar_t>(CblasTrans, CblasTrans, rank_, rank_, dim_, (scalar_t)1, M, weight_, scalar_t(0), R);
    for(uint32_t i = 0; i < rank_; i++) { R[i * rank_ + i] -= scalar_t(1); }
    cblas_gemm<scalar_t>(CblasNoTrans, CblasNoTrans, rank_, dim_, rank_, lambda * scalar_t(2), R, weight_, (scalar_t)1, G_sqr);
    cblas_axpby<scalar_t>(rtd, scalar_t(1) - beta1, G_sqr, beta1, M_adam);
    cblas_pow<scalar_t>(rtd, G_sqr, scalar_t(2), G_sqr);
    cblas_axpby<scalar_t>(rtd, scalar_t(1) - beta2, G_sqr, beta2, V_adam);
    cblas_axpby<scalar_t>(rtd, scalar_t(1) / (scalar_t(1) - beta2), V_adam, scalar_t(0), G_sqr);
    cblas_sqr<scalar_t>(rtd, G_sqr, G_sqr);
    cblas_add_scalar<scalar_t>(rtd, G_sqr, eps, G_sqr);
    cblas_div<scalar_t>(rtd, M_adam, G_sqr, G_sqr);
    cblas_axpy<scalar_t>(rtd, -eta / (scalar_t(1) - beta1), G_sqr, weight_);
    if(! quiet_) { std::cout<<"Iteration "<<iter<<" Loss "<<loss/(scalar_t)batch_size<<"\n"; }
    iter++;
  }
  delete [] M_adam;
  delete pairs;
}

template<typename scalar_t>
void WARCA<scalar_t>::train(const scalar_t* data, const uint32_t *labels,uint32_t num_data, 
                         uint32_t dim_data, uint32_t rank, scalar_t lambda, scalar_t eta, 
                         uint32_t max_iter, uint32_t batch_size, uint32_t max_sampling,
                         KernelType kernel_type, scalar_t gamma, uint32_t degree, scalar_t coeff) {
  if(weight_ != NULL) { delete [] weight_; }
  scalar_t* kernel_matrix = NULL;
  SVs_ = NULL;
  kernel_type_ = kernel_type;
  rank_ = rank;
  gamma_ = gamma;
  degree_ = degree;
  coeff_ = coeff;
  /*
    Pre computed kernels, "data" is then a kernel matrix
   */
  if(kernel_type_ == PRE_COMP) {
    dim_ = num_data;
    weight_ = new scalar_t [rank_ * dim_];
    train_adam_dual(data, labels, lambda, eta,  max_iter, batch_size, max_sampling);
    return;
  }
  if(kernel_type_ != LIN) {
    kernel_matrix = new scalar_t [num_data * num_data];
    dim_ = num_data;
    num_svs_ = num_data;
    dim_svs_ = dim_data;
    SVs_ = new scalar_t [num_svs_ * dim_svs_];
    cblas_copy<scalar_t>(num_data * dim_data, data, SVs_);
  }
  else {dim_ = dim_data; }
  weight_ = new scalar_t [rank_ * dim_];
  switch(kernel_type_) {
    case LIN:
      throw warca_error("Currently not supported,  it will be comming soon\n");
      break;
    case LINK:
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, num_data,
                      dim_data, scalar_t(1), data, data, scalar_t(0), kernel_matrix);
      break;
    case POLYNOMIAL:
      compute_polynomial_kernel(data, kernel_matrix, num_data,  dim_data);
      break;
    case RBF:
      compute_rbf_kernel(data, kernel_matrix, num_data,  dim_data);
      break;
    case CHI2RBF:
      compute_chi2rbf_kernel(data, kernel_matrix, num_data, dim_data);
      break;
    default:
      throw warca_error("Unrecognized kernel type\n");
      break;
  }
  if(kernel_type != LIN) { train_adam_dual(kernel_matrix, labels, lambda, eta,  max_iter, batch_size, max_sampling); }
  else { throw warca_error("Currently not supported,  it will be comming soon\n");  }
  if(kernel_matrix != NULL) { delete [] kernel_matrix; }
}



template<typename scalar_t>
void WARCA<scalar_t>::predict(const scalar_t* data, scalar_t *embedding,  uint32_t num_data, uint32_t dim_data) {
  scalar_t* kernel_matrix = NULL;
   switch(kernel_type_) {
    case PRE_COMP:
      if(dim_data != dim_) {
        throw warca_error("The input size doesn't match\n");
      }
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, rank_, dim_, scalar_t(1), data, weight_, scalar_t(0), embedding);
    case LINK:
      kernel_matrix = new scalar_t [num_data * num_svs_];
      if(dim_data != dim_svs_) {
        throw warca_error("The input size doesn't match with SVs\n");
      }
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, num_svs_,
                           dim_data, scalar_t(1), data, SVs_, scalar_t(0), kernel_matrix);
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, rank_, dim_, scalar_t(1), kernel_matrix, weight_, scalar_t(0), embedding);
      delete [] kernel_matrix;
      break;
    case POLYNOMIAL:
      if(dim_data != dim_svs_) {
        throw warca_error("The dimension of input doesn't match with SVs\n");
      }
      kernel_matrix = new scalar_t [num_data * num_svs_];
      compute_polynomial_kernel(data, SVs_,  kernel_matrix, num_data, num_svs_, dim_data);
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, rank_, dim_, scalar_t(1), kernel_matrix, weight_, scalar_t(0), embedding);
      delete [] kernel_matrix;
      break;
    case RBF:
      if(dim_data != dim_svs_) {
        throw warca_error("The dimension of input doesn't match with SVs\n");
      }
      kernel_matrix = new scalar_t [num_data * num_svs_];
      compute_rbf_kernel(data, SVs_,  kernel_matrix, num_data, num_svs_, dim_data);
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, rank_, dim_, scalar_t(1), kernel_matrix, weight_, scalar_t(0), embedding);
      delete [] kernel_matrix;
      break;
    case CHI2RBF:
      if(dim_data != dim_svs_) {
        throw warca_error("The dimension of input doesn't match with SVs\n");
      }
      kernel_matrix = new scalar_t [num_data * num_svs_];
      compute_chi2rbf_kernel(data, SVs_,  kernel_matrix, num_data, num_svs_, dim_data);
      cblas_gemm<scalar_t>(CblasNoTrans, CblasTrans, num_data, rank_, dim_, scalar_t(1), kernel_matrix, weight_, scalar_t(0), embedding);
      delete [] kernel_matrix;
      break;
    default:
      throw warca_error("Unrecognized kernel type\n");
      break;
  }
}


INSTANTIATE_CLASS(WARCA);

}


 
