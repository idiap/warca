# WARCA
WARCA is a simple and fast algorithm for metric learning. It can do linear metric learning as well as non-linear metric learning through kernels. This library provides an efficient and easy to use implementation of WARCA with c++ and matlab interface. It supports both double and single precision arithmetic.
This library is currently tested only on linux platforms using GNU C++ compiler. It requires GNU C++ compiler with c++11 support and a good blas implementation like openblas or atlas.

## Compilation of c++ interface & data format

Make sure that you have GNU c++ compiler with C++11 support and a blas installation (preferably openblas).  Execute make to build warca_train_*_precision and warca_prediction_*_precision.  
The data format of training and testing feature file is comma seperated values where each line indicates a data point. That is number of lines indicates the number of data points and number of columns indicates the data dimension.
Label file is a single column file with number of lines indicates the number of points and the value indicates the class the point belongs to.


## warca_train_{single, double}_precision.bin Usage

Usage: warca_train_{single, double}_precision [options] training_feature_file training_label_file [model_file]  
options:  
-r rank : set the dimension of projected space (default 2)    
-k kernel_type : set type of kernel function (default 2)    
  0 -- linear: u'*v\n"  
  1 -- linear kernel but the model is trained in kernel_space  
  2 -- polynomial: (gamma*u'*v + coef0)^degree  
  3 -- radial basis function: exp(-gamma*|u-v|^2)  
  4 -- chi2rbf radial basis function: exp(-gamma * chi2(u, v)^2)  
  5 -- precomputed kernel (kernel values in training\_feature\_file)  
-d degree : set degree in kernel function (default 3)  
-g gamma : set gamma in kernel function (default 1/num_features)  
-e eta : learning rate of SGD algorithm (default 1)  
-l lambda: Regularizer strength (default 1e-2)  
-i max_iter : Maximum number of SGD iterations(default 2000)  
-b batch_size : set batch size (default 512)  
-m max_sampling : Maximum number of sampling in WARP loss (default 512)  
-s seed : Seed of the random number generator (default 1)  
-q : quiet mode (no verbose)  

### Example

```bash
./warca_train_single_precision.bin -r 40 -k 4 -g 0.01 -e 0.1 -l 1e-4 -i 2000 -b 512 -m 512 -s 1 -q  ./data/toy_train_features.txt ./data/toy_train_labels.txt toy.model
```
Executing the above command trains a warca model, with rank of the embedding=40, with chi2rbf kernel, learning rate(eta) = 0.1, regularizer strength(lambda)=1e-4, maximum SGD updates (max_iter)=2000, batchsize=512 and max_sampling=512
on the toy dataset distributed along with the code. It also saves the model as toy.model in the current directory.


## warca_predict_{single, double}_precision.bin Usage

Usage: warca_predict_{single, double}_precision.bin feature_file model_file result_file

### Example

```bash
warca_predict_single_precision.bin  ./data/toy_test_features.txt ./toy.model ./toy_test_warca_embedding.txt
```

Executing the above command will compute the embedding on the toy_features using the warca model saved under ./toy.model and saves the computed embedding to ./toy_test_embedding.txt

## Compilation and usage of matlab interface

A matlab interface is provided under the directory matlab. The interface can be compiled by executing make.m  in the matlab console. Please note that you will require a mex compiler with c++11 support and a blas installation. 

**Currently the matlab interface is available only for pre-computed kernels  and linear warca** 

A demo of the matlab interface usage is given in examples/demo_VIPeR.m

### Issues with code
Please contact me at josevancijo@gmail.com

