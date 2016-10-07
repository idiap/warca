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

#include "warca_utils.hpp"
using namespace warca;
int main(int argc, char* argv[]) {
  if(argc != 4) { exit_warca_predict_with_help(); }
  std::string feature_filename = argv[1];
  std::string model_filename = argv[2];
  std::string result_filename = argv[3];
  uint32_t num_data, dim_data;
  float* data = NULL;
  read_features<float>(feature_filename,  &data, num_data, dim_data);
  WARCA<float> warca_predictor(model_filename.c_str());
  uint32_t rank = warca_predictor.rank();
  float* embedding = new  float [num_data * rank];
  warca_predictor.predict(data, embedding, num_data, dim_data);
  write_features(result_filename, embedding, num_data, rank);
  delete [] embedding;
  delete [] data;
}
