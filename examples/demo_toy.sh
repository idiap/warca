: 'This is a small demo script illustrating the  usage of warca library.  
It learns a warca embedding on the toy dataset given in data directory using chi2rbf kernel and then computes the embedding on both the data used for learning as well as on a given test.

We have also included a python implementation of tSNE visualization tool to visualize the space learned by WARCA. Please see the tSNE folder on how to compile the python extensions.
If you have successfully managed to run this script then you will get 4 pdf scatter plots and you can see on the plots that WARCA brings together the points of the samle class and pushes apart points from different 
class both on training (quite visible on training set) and test set.  Please note that this illustration is a zero-shot scenario that is in the given toy example the training and test classes are disjoint.
'

../warca_train_single_precision.bin -r 100 -k 4 -g 0.01  -e 0.1 -l 1e-4 -i 2000 -b 512 -m 512    ../data/toy_train_features.txt ../data/toy_train_labels.txt ../results/toy.model
../warca_predict_single_precision.bin  ../data/toy_train_features.txt ../results/toy.model  ../results/toy_train_warca_embed.txt
../warca_predict_single_precision.bin  ../data/toy_test_features.txt ../results/toy.model  ../results/toy_test_warca_embed.txt

python ./tSNE/tSNE.py ../data/toy_train_features.txt ../data/toy_train_labels.txt chi2
python ./tSNE/tSNE.py ../data/toy_test_features.txt ../data/toy_test_labels.txt  chi2

python ./tSNE/tSNE.py ../results/toy_train_warca_embed.txt  ../data/toy_train_labels.txt euc
python ./tSNE/tSNE.py ../results/toy_test_warca_embed.txt  ../data/toy_test_labels.txt euc
