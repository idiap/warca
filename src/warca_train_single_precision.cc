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
int main(int argc, char *argv[]) {
  std::string feature_filename;
  std::string label_filename;
  std::string model_filename;
  WARCATrainParameters<float> param;
  parse_command_line<float>(argc, argv, feature_filename, label_filename,
                            model_filename, param);
  uint32_t num_data, dim_data;
  float *data = NULL;
  uint32_t *labels = NULL;
  read_problem<float>(feature_filename, label_filename, &data, &labels,
                      num_data, dim_data);
  WARCA<float> warca_trainer(param.seed, param.quiet);
  warca_trainer.train(data, labels, num_data, dim_data, param.rank,
                      param.lambda, param.eta, param.max_iter, param.batch_size,
                      param.max_sampling, param.kernel_type, param.gamma,
                      param.degree, param.coeff);
  warca_trainer.save_model(model_filename.c_str());
  delete[] data;
  delete[] labels;
}
