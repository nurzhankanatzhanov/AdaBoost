% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)

load zip.train;
fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,(Y-2),'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200, 1, 0, 0);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

% test error stuff
load zip.test;
testing_subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
testingY = testing_subsample(:,1);
testingX = testing_subsample(:,2:257);
% learning on the training data
testing_ct = fitctree(X,(Y-2));
% learn a hypothesis
h = predict(testing_ct, testingX);
errors = h ~= (testingY - 2);
errors_sum = sum(errors);
N = size(testingX, 1);
single_tree = errors_sum/N;
fprintf('\nTESTING errors for 1vs3:\n');
fprintf('The testing error of single decision tree is %.4f\n', single_tree);
testing_bee = BaggedTrees(testingX, testingY, 200, 1, testingX, testingY);
fprintf('The testing error of 200 bagged decision trees is %.4f\n', testing_bee);

% 3vs5
load zip.train
fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample_3v5 = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_3v5 = subsample_3v5(:,1);
X_3v5 = subsample_3v5(:,2:257);
ct_3v5 = fitctree(X_3v5,(Y_3v5-4),'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct_3v5.kfoldLoss);
bee_3v5 = BaggedTrees(X_3v5, Y_3v5, 200, 3, 0, 0);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee_3v5);

load zip.test
testing_subsample_3v5 = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
testingY_3v5 = testing_subsample_3v5(:,1);
testingX_3v5 = testing_subsample_3v5(:,2:257);
% learning on the training data
testing_ct_3v5 = fitctree(X_3v5,(Y_3v5-4));
% learn a hypothesis
h_3v5 = predict(testing_ct_3v5, testingX_3v5);
errors_3v5 = h_3v5 ~= (testingY_3v5-4);
errors_sum_3v5 = sum(errors_3v5);
N_3v5 = size(testingX_3v5, 1);
single_tree_3v5 = errors_sum_3v5/N_3v5;
fprintf('\nTESTING errors for 3vs5:\n');
fprintf('The testing error of single decision tree is %.4f\n', single_tree_3v5);
testing_bee_3v5 = BaggedTrees(testingX_3v5, testingY_3v5, 200, 3, testingX_3v5, testingY_3v5);
fprintf('The testing error of 200 bagged decision trees is %.4f\n', testing_bee_3v5);
