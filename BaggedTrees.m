function [ oobErr ] = BaggedTrees( X, Y, numBags, option, testX, testY )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%       option: 1 for 1vs3 problem and 3 for 3vs5 problem
%       testX: optional matrix of testing data (0 if unused - training)
%       testY: option vector of classes of the testing examples (0 if
%       unused - training)
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function
    
    % switch all cases to -1 or +1 based on the case
    if option == 1
        Y = Y - 2;
        testY = testY - 2;
    elseif option == 3
        Y = Y - 4;
        testY = testY - 4;
    end
    
    baggingSize = size(X,1);
    OOB = zeros(numBags, 1);
    
    % initializing variables based on train/test modes
    if testX == 0
        G = zeros(baggingSize, numBags);
        N = baggingSize;
    else
        testSize = size(testX, 1);
        G = zeros(testSize, numBags);
        N = testSize;
    end
    
    for t=1:numBags
        % from lecture - repeatedly uniformly sample N points from D with replacement
        [data, index] = datasample(1:baggingSize, baggingSize);
        
        if testX == 0
            input = X(data, :);
            output = Y(index);
            % need to find points not in data(m) so that we don't train g(m)
            % idea from https://www.mathworks.com/matlabcentral/answers/39735-find-elements-in-one-array-not-in-another
            unique = setdiff(1:baggingSize, data, 'sorted');
            uniqueX = X(unique, :);
            
            % keeping cross-val option off so that we get max-depth tree
            ct = fitctree(input, output);
            
            % learning hypotheses
            G(unique, t) = predict(ct, uniqueX);
        else
            input = testX(data, :);
            output = testY(index);
            uniqueX = testX;
            
            % keeping cross-val option off so that we get max-depth tree
            ct = fitctree(input, output);
            
            % learning hypotheses
            G(:, t) = predict(ct, uniqueX);
        end
        
        % get column vector of every element in the 1:t columns (increasing with t)
        aggrG = G(:,1:t);
        % aggregated sums
        sum_aggrG = sum(aggrG, 2);
        % next need to know the error b/w G(x_n) and y_n
        % in this case we need BINARY error for classification
        errors = sign(sum_aggrG) ~= Y;
        % sum of all errors from n=1 to N
        errors_sum = sum(errors);
        % taking average over N, the size of our bootstrapped aggregating    
        OOB(t) = 1/N * errors_sum;
    end
    
    % report out-of-bag error (last element)
    oobErr = OOB(numBags);
    
    % plotting OOB Error as a function of the number of bags from 1 to
    % numBags
    if testX == 0
        figure;
        hold on
        scatter(1:numBags, OOB, 10, 'd', 'red', 'filled');
        line(1:numBags, OOB);
        ylabel("Out-Of-Bag Error");
        xlabel("# of bags");

        if option == 1
            title("One (1) vs. Three (3) Problem");
        elseif option == 3
            title("Three (3) vs. Five (5) Problem");
        end
    end
end
