load zip.train;
fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);

% test error stuff
load zip.test;
testing_subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
testingY = testing_subsample(:,1);
testingX = testing_subsample(:,2:257);

[train_err, test_err] = AdaBoost(X, Y, testingX, testingY, 200, 1);

fprintf('The AdaBoost training error is %.4f\n', train_err);
fprintf('The AdaBoost testing error is %.4f\n', test_err);


%%%%%% 3vs5 problem %%%%%%%


load zip.train
fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample_3v5 = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_3v5 = subsample_3v5(:,1);
X_3v5 = subsample_3v5(:,2:257);

load zip.test
testing_subsample_3v5 = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
testingY_3v5 = testing_subsample_3v5(:,1);
testingX_3v5 = testing_subsample_3v5(:,2:257);

[train_err_3v5, test_err_3v5] = AdaBoost(X_3v5, Y_3v5, testingX_3v5, testingY_3v5, 200, 3);

fprintf('The AdaBoost training error is %.4f\n', train_err_3v5);
fprintf('The AdaBoost testing error is %.4f\n', test_err_3v5);