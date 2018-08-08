clear global; clc

load('spamData.mat');

numFeatures = 57;
numTrain = length(Xtrain);
numTest = length(ytest);
numTotal =  numTrain + numTest;

%size of eta variation when doing line search
eta_size = 0.05;

%decide when to stop the loop to minimize NLL (in % of NLL)
threshold = 0.001;

lambda_array = [1:10,15:5:100]';

error_count = zeros(length(lambda_array), 1);
error_rate = zeros(length(lambda_array), 1);
error_count_train = zeros(length(lambda_array), 1);
error_rate_train = zeros(length(lambda_array), 1);

%%this counts the speed of convergence for each lambda value
while_loop_cnt = zeros(length(lambda_array), 1);

%%pre-processing step
[Z_Xtest, Z_Xtrain] = PreProcess.logTransform(Xtest, Xtrain);
%%

%concat the w0 "feature"
w0 = ones(numTrain,1);
Z_Xtrain_new = horzcat(w0,Z_Xtrain);

%%
%MAIN LOOP STARTS HERE
for n = 1:length(lambda_array)
    
    cost_func = 0;
    cost_func_next = 0;
    step_diff_percent = 100;
    while_loop_cnt(n) = 0;
    %%
    %newton's method starts here
    w = zeros(numFeatures+1, 1);
    w_next = zeros(numFeatures+1, 1);

    mu = zeros(numTrain,1);
    mu_next = zeros(numTrain,1);

    %to calculating mu(i) - first iteration
    for i = 1:numTrain
        mu(i) = 1/(1 + exp(-transpose(w) * transpose(Z_Xtrain_new(i,:))));
    end

    %calculating the cost function for w - first iteration
    for i = 1:numTrain
         cost_func = cost_func - (ytrain(i)*log(mu(i)) + ((1-ytrain(i))*log(1-mu(i))));
    end

    w_reg = w;
    w_reg(1,1) = 0;

    cost_func = cost_func +(0.5 * lambda_array(n,1) * transpose(w_reg) * w_reg);
    %%
    
    while cost_func_next < cost_func && step_diff_percent > 0.01

        %calculating mu(i)
        for i = 1:numTrain
            mu(i) = 1/(1 + exp(-transpose(w) * transpose(Z_Xtrain_new(i,:))));
        end

        cost_func = 0;
        %calculating the cost function for w
        for i = 1:numTrain
            cost_func = cost_func - (ytrain(i)*log(mu(i)) + ((1-ytrain(i))*log(1-mu(i))));
        end

        w_reg = w;
        w_reg(1,1) = 0;

        cost_func = cost_func + (0.5 * lambda_array(n,1) * transpose(w_reg) * w_reg);

        %calculating g
        g_reg = lambda_array(n,1) * w;
        g_reg(1,1) = 0;

        g = (transpose(Z_Xtrain_new)*(mu - ytrain)) + g_reg;

        %calculating S
        S = zeros(numTrain);
        for i = 1:numTrain
            S(i,i) = mu(i) * (1 - mu(i));
        end

        %calculating H
        H_reg = lambda_array(n,1) * eye(numFeatures+1);
        H_reg(1,1) = 0;

        H = (transpose(Z_Xtrain_new) * S * Z_Xtrain_new) + H_reg; 

        %calculating dk
        dk = H\(-g);

        %%starting line search to find best eta  
        eta_choices =  zeros((1/eta_size)+1, 1);
        cost_func_choices = zeros((1/eta_size)+1, 1);
        mu_try = zeros(numTrain,1);

        array_idx = 1; %keep track of our eta selection array

        %%
        %loop through eta = 1 to eta = 0, with step size of eta_size
        for eta = 1: -eta_size: 0

            w_try = w + eta*dk;
            eta_choices(array_idx) = eta;

            %calculating mu_next(i)
            for i = 1:numTrain
                mu_try(i) = 1/(1 + exp(-transpose(w_try) * transpose(Z_Xtrain_new(i,:))));
            end

            %calculating the cost function for w_try
            for i = 1:numTrain
                cost_func_choices(array_idx) = cost_func_choices(array_idx) - (ytrain(i)*log(mu_try(i)) + ((1-ytrain(i))*log(1-mu_try(i))));
            end

            w_reg = w;
            w_reg(1,1) = 0;

            cost_func_choices(array_idx) = cost_func_choices(array_idx) + (0.5 * lambda_array(n,1) * transpose(w_reg) * w_reg);

            array_idx = array_idx + 1;
        end
        %%

        %%pick best eta (lowest cost_func in the array of choices), and finalize w_next
        [cost_func_next, eta_idx] = min(cost_func_choices);

        w_next = w + eta_choices(eta_idx)*dk;

        %set w_next as w, prepare for next while-loop iteration
        w = w_next;
        step_diff_percent = abs(cost_func_next - cost_func)/abs(cost_func);
        while_loop_cnt = while_loop_cnt + 1;

    end

    %%
    %concat the w0 "feature" to the test set
    w0 = ones(numTest,1);
    Z_Xtest_new = horzcat(w0,Z_Xtest);

    %%
    %testing for test set
    spam_result = zeros(numTest,1);

    for i = 1:numTest

        %%testing for class 1 - i.e. spam
        %p_y1 = 1/(1 + exp(-(w0(i) + transpose(w) * transpose(Z_Xtest_new(i,:)))));
        p_y1 = 1/(1 + exp(-transpose(w) * transpose(Z_Xtest_new(i,:))));
        
        if p_y1 > 0.5
            spam_result(i) = 1;
        end    
    end

    for i = 1:numTest

        if spam_result(i) ~= ytest(i)
            error_count(n) = error_count(n) + 1;
        end

    end

    error_rate(n) = error_count(n)/numTest*100;

    %%
    %testing for training set
    w0 = ones(numTrain,1);
    spam_result_train = zeros(numTrain,1);

    for i = 1:numTrain

        %%testing for class 1 - i.e. spam
        %p_y1_train = 1/(1 + exp(-(w0(i) + transpose(w) * transpose(Z_Xtrain_new(i,:)))));
         p_y1_train = 1/(1 + exp(-transpose(w) * transpose(Z_Xtrain_new(i,:))));

        if p_y1_train > 0.5
            spam_result_train(i) = 1;
        end 
    end

    for i = 1:numTrain

        if spam_result_train(i) ~= ytrain(i)
            error_count_train(n) = error_count_train(n) + 1;
        end
    end

    error_rate_train(n) = error_count_train(n)/numTrain*100;

end
%%
plot(lambda_array,error_rate,lambda_array,error_rate_train)

xlabel('lambda');
ylabel('Error rates (%)');
title('Q3(log-Transformed): Plot of error rates vs. lambda'); 
legend('Test set', 'Training set','Location', 'southeast');

hold on;

plot(lambda_array(1),error_rate(1) ,'bx');
plot(lambda_array(10), error_rate(10),'bx');
plot(lambda_array(28), error_rate(28),'bx');

plot(lambda_array(1),error_rate_train(1),'rx');
plot(lambda_array(10), error_rate_train(10), 'rx');
plot(lambda_array(28), error_rate_train(28),'rx');

hold off;

fprintf('log-Transformed data-set:\n');
fprintf('Test error for lambda = 1 : %.4f\n',error_rate(1));
fprintf('Test error for lambda = 10 : %.4f\n',error_rate(10));
fprintf('Test error for lambda = 100 : %.4f\n',error_rate(28));

fprintf(' \n');

fprintf('Training error for lambda = 1 : %.4f\n',error_rate_train(1));
fprintf('Training error for lambda = 10 : %.4f\n',error_rate_train(10));
fprintf('Training error for lambda = 100 : %.4f\n',error_rate_train(28));
