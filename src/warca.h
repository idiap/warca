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

#ifndef _WARCA_H_
#define _WARCA_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include "random.h"

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

namespace warca {

class warca_error : public std::exception {
public:
    warca_error(const std::string& msg) : msg_(msg){}
    ~warca_error() throw() {}
    const char* what() const throw() { return msg_.c_str(); }
private:
    std::string msg_;
};

class SimilarPairs {
  uint32_t* pairs_;
  uint32_t count_;
public:
  SimilarPairs(const uint32_t *y, const uint32_t n) {
    count_ = 0;
    for(uint32_t i=0; i<n; i++) 
      for(uint32_t j=i+1; j<n; j++) 
	if(y[i] == y[j])
	  count_++;
    pairs_ = new uint32_t [count_ * 2];
    uint32_t m = 0;
    for(uint32_t i=0; i<n; i++) { 
      for(uint32_t j=i+1; j<n; j++) { 
	if(y[i] == y[j]) {
	  pairs_[m * 2] = i;
	  pairs_[m * 2 + 1] = j;
	  m++;
	}
      }
    }
  }
  inline void sample_pair(uint32_t *i, uint32_t *j, Prand *rnd) {
    uint32_t k = rnd->randi(0, count_);
    *i = pairs_[k * 2];
    *j = pairs_[k * 2 + 1];
  }
  ~SimilarPairs() {
    delete [] pairs_;
  }
};

/*
  Currently supported kernel types. LIN and LINK are both linear kernels. LIN indicates that the model should be trained in the input space and LINK indicates that the model should be trained in
  the kernel space
*/
enum KernelType{LIN=0, LINK, RBF, POLYNOMIAL, CHI2RBF, PRE_COMP};

template<typename scalar_t>
class WARCA {
  KernelType kernel_type_;
  Prand *rnd_;
  //Parameter matrix
  scalar_t *weight_;
  //Support vectors (training points), valid only for non-linear kernels. For linear it will be null.
  scalar_t *SVs_;
  //The number of columns of W, for kernel WARCA it will be the number of training points and for linear warca it is the data dimension;
  uint32_t dim_;
  //The number of rows of W, indicates the dimension of low rank projection;
  uint32_t rank_;
  //The number of support vectors, valid only for non-linear kernels (For kernel warca num_svs_ and dim_ will be same)
  uint32_t num_svs_;
  //Dimnesion of support vectors, valid only for non-linear kernels
  uint32_t dim_svs_;
  //Kernel_width
  scalar_t gamma_;
  //Degree of the polynomial kernel
  scalar_t degree_;
  //bias of the polynomial kernel
  scalar_t coeff_;
  //verbose
  bool quiet_;
  
  void compute_chi2rbf_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,  uint32_t num_x, uint32_t num_y, uint32_t dim);
  void compute_chi2rbf_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix, uint32_t num_x, uint32_t dim);
  void compute_polynomial_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,
                                 uint32_t num_x, uint32_t num_y, uint32_t dim);
  void compute_polynomial_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix, uint32_t num_x, uint32_t dim);
  void compute_rbf_kernel(const scalar_t* x_matrix, const scalar_t* y_matrix, scalar_t* kernel_matrix,
                          uint32_t num_x, uint32_t num_y, uint32_t dim);
  void compute_rbf_kernel(const scalar_t* x_matrix, scalar_t* kernel_matrix, uint32_t num_x, uint32_t dim);
  scalar_t l2dist(const scalar_t* x, const scalar_t* y, uint32_t dim);
  scalar_t warp_gradient_dual(const scalar_t* M, const uint32_t* labels, scalar_t *G,
                              uint32_t batch_size, uint32_t max_sampling, SimilarPairs *pairs);
  void train_adam_dual(const scalar_t *kernel_matrix, const uint32_t *labels, scalar_t lambda,
                       scalar_t eta, uint32_t max_iter, uint32_t batch_size, uint32_t max_sampling);
 public:
  WARCA(scalar_t seed, bool quiet);
  WARCA(scalar_t seed);
  WARCA(const char* filename, scalar_t seed);
  WARCA(const char* filename);
  WARCA();
  WARCA(bool quiet);
  ~WARCA() {
    delete rnd_;
    if(weight_ != NULL) { delete [] weight_; }
    if(SVs_ != NULL) { delete [] SVs_; }
  }
  void save_model(const char* filename);
  void load_model(const char* filename);
  void train(const scalar_t* data, const uint32_t *labels,uint32_t num_data, 
             uint32_t dim_data, uint32_t rank, scalar_t lambda, scalar_t eta,  uint32_t max_iter,
             uint32_t batch_size, uint32_t max_sampling, KernelType kernel_type,
             scalar_t gamma, uint32_t degree, scalar_t coeff);
  void predict(const scalar_t* data, scalar_t *data_proj, uint32_t num_data, uint32_t dim_data);
  inline uint32_t dim() { return dim_; }
  inline uint32_t rank() { return rank_; }
  inline uint32_t num_svs() { return num_svs_; }
  inline uint32_t dim_svs() { return dim_svs_; }
  inline scalar_t gamma() { return gamma_; }
  inline KernelType kernel_type() { return kernel_type_; }
  inline const scalar_t* weight() { return (const scalar_t*)weight_; }
  inline const scalar_t* SVs() { return (const scalar_t*)SVs_; }
};

}
#endif
