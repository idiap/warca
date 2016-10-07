This is a tool for visualizing points in high dimensional spaces in 2d. It supports arbitrary metric over the data (see tsne.py)
 
To compile fast_functions.pyx execute the following

python setup.py build_ext --inplace

Please note that this is a toy implementation and do not use if you have more than 5000 points to visualize however it scales to very large dimensions.