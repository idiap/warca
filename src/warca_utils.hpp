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

#ifndef _WARCA_UTILS_HPP_
#define _WARCA_UTILS_HPP_

#include "warca.h"
#include <cstdio>
#include <cstring>
#include <sstream>
#include <fstream>
#include <string>
#include <cassert>
#include <iostream>
using namespace warca;

template<typename scalar_t>
struct WARCATrainParameters {
  uint32_t rank;
  KernelType kernel_type;
  uint32_t degree;
  scalar_t coeff;
  scalar_t gamma;
  scalar_t eta;
  scalar_t lambda;
  uint32_t max_iter;
  uint32_t batch_size;
  uint32_t max_sampling;
  scalar_t seed;
  bool quiet;
};



void exit_warca_train_with_help() {
  throw warca_error(
      "Usage: warca_train [options] feature_file label_file [model_file]\n"
      "options:\n"
      "-r rank : set the dimension of projected space (default 2)\n"
      "-k kernel_type : set type of kernel function (default 2)\n"
      "0 -- linear: u'*v\n"
      "1 -- linear kernel but the model is trained in kernel_space\n"
      "2 -- polynomial: (gamma*u'*v + coef0)^degree\n"
      "3 -- radial basis function: exp(-gamma*|u-v|^2)\n""-d degree : set degree in kernel function (default 3)\n"
      "4 -- chi2 radial basis function: exp(-gamma * chi2(u, v)^2)\n"
      "5 -- precomputed kernel (kernel values in training_set_file)\n"
      "-d degree : degree of the polynomial kernel (default 3)\n"
      "-c coeff : bias of polynomial kernel (default 0)\n"
      "-g gamma : set gamma in kernel function (default 1/num_features)\n"
      "-e eta : learning rate (efault 0)\n"
      "-l  lambda: Regularizer strength (default 1e-2)\n"
      "-i max_iter : Number of SGD iterations(default 2000)\n"
      "-b batch_size : set batch size (default 512)\n"
      "-m max_sampling : Maximum number of sampling in WARP loss (default 512)\n"
      "-s seed : Seed for random number generator (default 1)\n"
      "-q : quiet mode (no outputs)\n"
         );
  exit(1);
}

void exit_warca_predict_with_help() {
  throw warca_error( "Usage: warca_predict feature_file model_file result_file\n");
  exit(1);
}


template<typename scalar_t>
void parse_command_line(uint32_t argc, char **argv, std::string& feature_filename, std::string& label_filename, std::string& model_filename, WARCATrainParameters<scalar_t> &param) {
  // default values
  uint32_t i;
  param.rank = 2;
  param.kernel_type = CHI2RBF;
  param.degree = 3;
  param.coeff = 0;
  param.gamma = scalar_t(0);// 1/num_features
  param.eta = scalar_t(1);
  param.lambda = scalar_t(1e-2);
  param.max_iter = 2000;
  param.batch_size = 512;
  param.max_sampling = 512;
  param.quiet = false;
  // parse options
  for(i = 1; i < argc; i++)  {
    if(argv[i][0] != '-') break;
    if(++i >= argc) { exit_warca_train_with_help(); }
    switch(argv[i-1][1]) {
      case 'r':
        param.rank = uint32_t(atoi(argv[i]));
        break;
      case 'k':
        param.kernel_type = KernelType(atoi(argv[i]));
        break;
      case 'd':
        param.degree = uint32_t(atoi(argv[i]));
        break;
      case 'c':
        param.coeff = scalar_t(atof(argv[i]));
        break;
      case 'g':
        param.gamma = scalar_t(atof(argv[i]));
        break;
      case 'e':
        param.eta = scalar_t(atof(argv[i]));
        break;
      case 'l':
        param.lambda = scalar_t(atof(argv[i]));
        break;
      case 'i':
        param.max_iter = uint32_t(atoi(argv[i]));
        break;
      case 'b':
        param.batch_size = uint32_t(atoi(argv[i]));
        break;
      case 'm':
        param.max_sampling = uint32_t(atoi(argv[i]));
        break;
      case 's':
        param.seed = scalar_t(atof(argv[i]));
        break;
      case 'q':
        param.quiet = true;
        i--;
        break;
      default:
        throw warca_error("Unknown option \n");
        exit_warca_train_with_help();
    }
  }
  // determine filenames
  if(i >= argc - 1) { exit_warca_train_with_help(); }
  feature_filename = argv[i];
  i++;
  label_filename = argv[i];
  if(i < argc-1)
    model_filename = argv[i+1];
  else {
    model_filename = "DefaultName.model";
  }
}

template<typename scalar_t>
void read_problem(const std::string feature_filename, const std::string label_filename, scalar_t** data, uint32_t **labels,  uint32_t &num_data, uint32_t &dim_data) {
  num_data = 0;
  dim_data = 0;
  std::ifstream fin(feature_filename.c_str());
  assert(fin.good());
  while(fin) {
    std::string s;
    if(!getline(fin, s, '\n')) break;
    num_data++;
    std::istringstream ss(s);
    std::string data;
    while(getline(ss, data, ',')) {
      dim_data++;
    }
  }
  dim_data /= num_data;
  fin.close();
  *data = new scalar_t [num_data * dim_data];
  scalar_t *x = *data;
  fin.open(feature_filename.c_str());
  size_t i = 0;
  while(fin) {
    std::string s;
    if(!getline(fin, s, '\n')) break;
    std::istringstream ss(s);
    std::string data;
    while(getline(ss, data, ',')) {
      std::istringstream sb(data);
      sb>>x[i];
      i++;
    }
  }
  fin.close();
  *labels = new uint32_t[num_data];
  uint32_t *y = *labels;
  fin.open(label_filename.c_str());
  assert(fin.good());
  i = 0;
  while(fin) {
    std::string s;
    if(!getline(fin, s, '\n')) break;
    std::istringstream ss(s);
    ss>>y[i];
    i++;
  }
  fin.close();
}

template<typename scalar_t>
void read_features(const std::string feature_filename, scalar_t** data, uint32_t &num_data, uint32_t &dim_data) {
  num_data = 0;
  dim_data = 0;
  std::ifstream fin(feature_filename.c_str());
  assert(fin.good());
  while(fin) {
    std::string s;
    if(!getline(fin, s, '\n')) break;
    num_data++;
    std::istringstream ss(s);
    std::string data;
    while(getline(ss, data, ',')) {
      dim_data++;
    }
  }
  dim_data /= num_data;
  fin.close();
  *data = new scalar_t [num_data * dim_data];
  scalar_t *x = *data;
  fin.open(feature_filename.c_str());
  size_t i = 0;
  while(fin) {
    std::string s;
    if(!getline(fin, s, '\n')) break;
    std::istringstream ss(s);
    std::string data;
    while(getline(ss, data, ',')) {
      std::istringstream sb(data);
      sb>>x[i];
      i++;
    }
  }
  fin.close();
}

template<typename scalar_t>
void write_features(const std::string feature_filename, const  scalar_t* data, uint32_t num_data, uint32_t dim_data) {
  std::ofstream fout(feature_filename.c_str());
  for(uint32_t i = 0; i < num_data; i++) {
    const scalar_t *data_i = data + i * dim_data;
    for(uint32_t j =0; j < dim_data - 1; j++) { fout<<data_i[j]<<","; }
    fout<<data_i[dim_data - 1]<<"\n";
  }
  fout.close();
}

#endif
