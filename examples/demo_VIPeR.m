addpath('../matlab');
dataset_name = 'VIPeR'
feaFile = ['../data/', dataset_name, '_feature.mat'];
num_classes = 632;
num_folds = 10;
num_ranks = 30;

load(feaFile, 'descriptors');
seed = 0;
rng(seed)
batch_size = 512;
rank = 100;
max_iter =  2000;
max_sampling = 512;
lambda =  1e-3;
eta = 0.1;
method.kernel = 'CHI2RBF';
method.gamma = 0.01;
galFea = descriptors(1 : num_classes, :);
probFea = descriptors(num_classes + 1 : end, :);
warca_models = cell(num_folds, 1);   
cmc = zeros(num_folds, num_ranks);
verbose = false;
%clear descriptors 
for nf = 1 : num_folds
    p = randperm(num_classes);
    galFea1 = galFea( p(1:num_classes/2), : );
    probFea1 = probFea( p(1:num_classes/2), : );
    X = [galFea1; probFea1];
    y = [(1:num_classes/2)'; (1:num_classes/2)'];
    [K_train, method] = compute_kernel(method, X);
    t0 = tic;
    W = warca_train(K_train, y, rank, lambda, eta, max_iter, batch_size, max_sampling,...
                    5, seed, verbose);
    trainTime = toc(t0);
    warca_models{nf} = W';
    clear galFea1 probFea1
    galFea2 = galFea(p(num_classes/2+1 : end), : );
    probFea2 = probFea(p(num_classes/2+1 : end), : );
    K_test_gal = compute_kernel(method, X, galFea2); 
    K_test_prob = compute_kernel(method, X, probFea2); 
    t0 = tic;
    xTestGal = warca_models{nf} * K_test_gal;
    xTestProb = warca_models{nf} * K_test_prob;
    xTest = [xTestGal, xTestProb];
    yTest = 1 : num_classes / 2;
    yTest = [yTest'; yTest'];
    idx_gal = 1 : num_classes / 2;
    idx_prob = num_classes / 2 + 1 : num_classes;
    cmc(nf, :) = compute_cmc(xTest, yTest, uint32(idx_gal),  uint32(idx_prob), num_ranks);
    matchTime = toc(t0);
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cmc(nf,[1,5,10,15,20]) * 100);
end
mean_cmc = mean(cmc);
plot(1 : num_ranks, mean_cmc);
model_name = [dataset_name, '_gamma_', num2str(method.gamma) '_reg_', num2str(lambda), '_', method.kernel,'_', num2str(rank),'_',num2str(eta),'_',num2str(batch_size),'.mat'];
save(['../results/', model_name], 'cmc', 'warca_models');
fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', mean_cmc([1,5,10,15,20]) * 100);
