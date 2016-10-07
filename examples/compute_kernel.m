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

function [K, kernel_params] = compute_kernel(kernel_params, X1, X2)
% Computes the kernel matrix of the data matrix X
% INPUTS
% kernel_params - structure indicating the kernel type and the kernel
% parameters.
% kernel_params.kernel - kernel type
% kernel_params.gamma - Width of the radial basis kernel
% kernel_params.coeff - Bias of the polynomial kernel
%kernel_params.degree - degree of the polynomial kernel
% X1  - [n1 x d] matrix of data points 'n' points in 'd' dimension
% X2  - [n2 x d] matrix of data points 'n' points in 'd' dimension
% OUTPUTS
%if(nargin == 2) { K  - [n1 x n1] kernel matrix }
%if(nargin == 3) { K  - [n1 x n2] kernel matrix }
% kernel_params The input structure
    if(nargin == 2)
        switch kernel_params.kernel
          case {'LINEAR'}% linear kernel
            K = X1 * X1';
          case {'POLYNOMIAL'}% Polynomial  kernel
            K = X1 * X1';
            if(~isfield(kernel_params, 'coeff'))
                kernel_params.coeff = 0.0;
            end
            if(~isfield(kernel_params, 'degree') || kernel_params.degree == 0)
                kernel_params.degree = 3;
            end
            K = K + kernel_params.coeff;
            K = K .^ kernel_params.degree;
          case {'RBF'}% RBF kernel
            K = X1 * X1';
            z = diag(K);
            K = 2 * K;
            K = bsxfun(@minus, K, z);
            K = bsxfun(@minus, K, z');
            if(~isfield(kernel_params, 'gamma') || kernel_params.gamma == 0)
                kernel_params.gamma = 1.0;
            end
            K =exp(K * kernel_params.gamma);
          case {'CHI2RBF'}% chi2 RBF kernel
            for i =1:size(X1,1)
                subp = bsxfun(@minus, X1(i,:), X1);
                subp = subp.^2;
                sump = bsxfun(@plus, X1(i,:), X1);
                K(i,:) = full(sum(subp./(sump+1e-10),2));
            end
            temp = triu(ones(size(K))-eye(size(K)))>0;
            temp = K(temp(:));
            [temp,~]= sort(temp);
            if(~isfield(kernel_params, 'gamma') || kernel_params.gamma == 0)
                kernel_params.gamma = 1.0;
            end
            K =exp( -K * kernel_params.gamma);
            clear subp sump;
        end
    end
    if(nargin == 3)
        switch kernel_params.kernel
          case {'LINEAR'}
            K = X1* X2';
          case {'POLYNOMIAL'}
            K  = X1* X2';
            if(~isfield(kernel_params, 'coeff'))
                kernel_params.coeff = 0.0;
            end
            if(~isfield(kernel_params, 'degree') || kernel_params.degree == 0)
                kernel_params.degree = 3;
            end
            K = K + kernel_params.coeff;
            K = K .^ kernel_params.degree;
          case {'RBF'}
            K  = -2 * X1* X2';
            x1 = sum(X1.^2, 2);
            x2 = sum(X2.^2, 2);
            bsxfun(@plus, K, x1);
            bsxfun(@plus, K, x2);
            if(~isfield(kernel_params, 'gamma') || kernel_params.gamma == 0)
                kernel_params.gamma = 1.0;
            end
            K  =exp(-K * kernel_params.gamma);
          case {'CHI2RBF'}
            for i =1:size(X2,1)
                subp = bsxfun(@minus, X2(i,:), X1);
                subp = subp.^2;
                sump = bsxfun(@plus, X2(i,:), X1);
                K(:, i) =  sum(subp ./ (sump + 1e-10), 2);
            end
            if(~isfield(kernel_params, 'gamma') || kernel_params.gamma == 0)
                kernel_params.gamma = 1.0;
            end
            K = exp(-K * kernel_params.gamma);
        end
    end
end
