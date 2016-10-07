%{
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
%}

function [w] = warca_train(data, labels, mrank, lambda, eta, max_iter, ...
                           batch_size, max_sampling, kernel_type, ...
                           seed, quiet)

[w] = warca_train_mex(double(data), uint32(labels), uint32(mrank), ...
                      double(lambda), double(eta),  uint32(max_iter), ...
                      uint32(batch_size), uint32(max_sampling), uint32(kernel_type), double(seed), quiet);
end
