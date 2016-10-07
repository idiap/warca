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
function [cmc] = compute_cmc(xTest, yTest, idx_gal, idx_prob, num_ranks)
    uy = unique(yTest);
    [num_trail, n] = size(idx_gal);
    assert(n == length(uy));
    R = zeros(num_trail, length(uy));
    for i=1:num_trail
        yTestProb = yTest(idx_prob(i,:)');
        yTestGal = yTest(idx_gal(i,:)');
        d = size(xTest, 1);
        D = eye(d); 
        x_a_test = xTest(:, idx_prob(i,:)');
        x_b_test = xTest(:, idx_gal(i,:)');
        N1 = size(x_a_test, 2);
        N2 = size(x_b_test, 2);
        f1 = repmat(diag(x_a_test'*D*x_a_test),[1,N2]);
        f2 = repmat(diag(x_b_test'*D*x_b_test)',[N1, 1]);
        f3 = x_a_test'*D*x_b_test;
        A =  f1+f2-2.0*f3;
        r = zeros(N1, 1);
        for j = 1:N1
            [~,IX_D] = sort(A(j,:));
            temp_idx = find(yTestGal(IX_D) == yTestProb(j));
            r(j) = temp_idx;
        end
        R(i, :)=r;
    end
    R = [R;R];
    [a, b] = hist(R' ,1:size(R,2));
    a = a./repmat(ones(1,size(a,2))*size(R,2), size(a,1),1);
    cmc = mean(a');
    cmc = cumsum(cmc);
    cmc = cmc(1:num_ranks);
end
