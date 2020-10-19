function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees, option )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
%   option: either 1 (for 1vs3) or 3 (for 3vs5)

    % switch all cases to -1 or +1 based on the case
    if option == 1
        y_tr = y_tr - 2;
        y_te = y_te - 2;
    elseif option == 3
        y_tr = y_tr - 4;
        y_te = y_te - 4;
    else
        error("Wrong input for option");
    end
    
    N = size(X_tr, 1);
    
    alpha = zeros(n_trees, 1);
    
    % initialize D_1(n) = 1/N for all n = 1, ..., N
    D = zeros(N,1);
    D(:) = 1.0 / N;
    
    % training collection
    training = ones(n_trees, N);
    testing = ones(n_trees, size(X_te, 1));
    
    trainSize = size(y_tr ,1);
    testSize = size(y_te, 1);
    
    for t = 1:n_trees
        % get decision stump from fitctree()
        % kept getting "No data remains in X and Y after removing
        % observations with zero and NaN weights" error, so added
        % "MaxNumSplits" parameter
        stump = fitctree(X_tr, y_tr, "SplitCriterion", "deviance", "MaxNumSplits", 1, "Weights", D);
        
        % learn g_t from the stump for both train and test sets
        g_t = predict(stump, X_tr);
        g_t_test = predict(stump, X_te);
        
        % calculate epsilon_t
        errors = g_t ~= y_tr;
        products = D .* errors;
        epsilon_t = sum(products);
        
        % calculate alpha_t
        alpha(t) = 1.0/2.0 * log((1-epsilon_t) / epsilon_t);
        
        % reweight rule
        D = D .* exp( (-1) * alpha(t) .* g_t .* y_tr);
        
        % adjusting for normalization constant
        Z = sum(D);
        D = D ./ Z;
        
        % collecting weak classifiers
        training(t, :) = g_t;
        testing(t, :) = g_t_test;
        
        % aggregation rule
        Gx = sign(sum(alpha .* training));
        Gx_test = sign(sum(alpha .* testing));
        
        train_err(t) = 1/trainSize * sum(Gx' ~= y_tr);
        test_err(t) = 1/testSize * sum(Gx_test' ~= y_te);
    end
    
    figure;
    hold on;
    plot(1:n_trees, train_err, 'Color', 'red');
    plot(1:n_trees, test_err, 'Color', [0 0.39 0]);
    ylabel("Training & Testing Error");
    xlabel("# of weak hypotheses");
    legend("Training Error", "Testing Error");
    
    if option == 1
        title("One (1) vs. Three (3) Problem - AdaBoost");
    elseif option == 3
        title("Three (3) vs. Five (5) Problem - AdaBoost ");
    end
    
    % report last value (what it converges to at point n_trees)
    train_err = train_err(n_trees);
    test_err = test_err(n_trees);
    
end

