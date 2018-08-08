clear global; clc

load('spamData.mat');

numFeatures = 57;

numTrain = length(Xtrain);
numTest = length(ytest);
numTotal =  numTrain + numTest;

%%pre-processing step
[B_Xtest, B_Xtrain] = PreProcess.binariZe(Xtest, Xtrain);
%%

%%find class prior pi_c: p(y=1|x,T)
N1 = 0; N = numTrain;

for i = 1:length(ytrain)
    if ytrain(i) == 1
       N1 = N1 + 1;
    end   
end

N0 = N - N1;
pi_c1 = N1/N;
pi_c0 = 1 - pi_c1;
%%

%%to count the number of alphas to keep track of
a = 0;
error_rate_index = 0;

while a <= 100
    error_rate_index = error_rate_index + 1;
    a = a + 0.5;
end;
%%

a = 0;
error_rate = zeros(error_rate_index,1);
error_rate_train = zeros(error_rate_index,1);

while a <= 100
    
    %find feature posterior: p(xj|y=1)
    posterior_c1 = zeros(numFeatures, 1);
    posterior_c0 = zeros(numFeatures, 1);
    
    alpha = a;
    beta = alpha;

    for i = 1:numFeatures
        Nj1 = 0;
        for j = 1:numTrain
            if B_Xtrain(j,i) == 1 && ytrain(j) == 1
              Nj1 = Nj1 + 1;
            end 
        end
       posterior_c1(i) = (Nj1+alpha)/(N1+alpha+beta);
    end

    for i = 1:numFeatures
        Nj0 = 0;
        for j = 1:numTrain
            if B_Xtrain(j,i) == 1 && ytrain(j) == 0
              Nj0 = Nj0 + 1;
            end 
        end
       posterior_c0(i) = (Nj0+alpha)/(N0+alpha+beta);
    end

    %%
    %testing on the test data set
    %%calculating for class y = 1

    log_y1 = zeros(numTest, 1);
    y1 = zeros(numTest, 1);

    for i = 1:numTest
        log_y1(i) = log(pi_c1);
        for j = 1:numFeatures
            if B_Xtest(i,j) == 1
                log_y1(i) = log_y1(i) + log(posterior_c1(j));
            elseif B_Xtest(i,j) == 0
                log_y1(i) = log_y1(i) + log((1 - posterior_c1(j)));
            end
        end
        y1 = exp(log_y1);
    end

    %%
    %%calculating for class y = 0
    log_y0 = zeros(numTest, 1);
    y0 = zeros(numTest, 1);

    for i = 1:numTest
        log_y0(i) = log(pi_c0);
        for j = 1:numFeatures
            if B_Xtest(i,j) == 1
                log_y0(i) = log_y0(i) + log(posterior_c0(j));
            elseif B_Xtest(i,j) == 0
                log_y0(i) = log_y0(i) + log((1 - posterior_c0(j)));
            end
        end
        y0 = exp(log_y0);  
    end

    %%normalize and count error
    y1_norm = zeros(numTest, 1);
    spam_result = zeros(numTest, 1);
    error_count = 0;

    for i = 1:numTest
        y1_norm(i) = y1(i)/(y1(i)+y0(i));

        if y1_norm(i) > 0.5
         spam_result(i) = 1;
        end

        if spam_result(i) ~= ytest(i)
            error_count = error_count + 1;
        end
    end

    error_rate(1 +(2*a)) = error_count/length(ytest)*100;
    
    %testing on the training data set
    %%calculating for class y = 1
    log_y1_train = zeros(numTrain, 1);
    y1_train = zeros(numTrain, 1);

    for i = 1:numTrain
        log_y1_train(i) = log(pi_c1);
        for j = 1:numFeatures
            if B_Xtrain(i,j) == 1
                log_y1_train(i) = log_y1_train(i) + log(posterior_c1(j));
            elseif B_Xtrain(i,j) == 0
                log_y1_train(i) = log_y1_train(i) + log((1 - posterior_c1(j)));
            end
        end
        y1_train = exp(log_y1_train);
    end

    %%
    %%calculating for class y = 0
    log_y0_train = zeros(numTrain, 1);
    y0_train = zeros(numTrain, 1);

    for i = 1:numTrain
        log_y0_train(i) = log(pi_c0);
        for j = 1:numFeatures
            if B_Xtrain(i,j) == 1
                log_y0_train(i) = log_y0_train(i) + log(posterior_c0(j));
            elseif B_Xtrain(i,j) == 0
                log_y0_train(i) = log_y0_train(i) + log((1 - posterior_c0(j)));
            end
        end
        y0_train = exp(log_y0_train);  
    end

    %%normalize and count error
    y1_norm_train = zeros(numTrain, 1);
    spam_result_train = zeros(numTrain, 1);
    error_count_train = 0;

    for i = 1:numTrain
        y1_norm_train(i) = y1_train(i)/(y1_train(i) + y0_train(i));

        if y1_norm_train(i) > 0.5
         spam_result_train(i) = 1;
        end

        if spam_result_train(i) ~= ytrain(i)
            error_count_train = error_count_train + 1;
        end
    end

    error_rate_train(1 +(2*a)) = error_count_train/length(ytrain)*100;
    
    a = a + 0.5;
end
%%

%to create an array of alpha values for plotting
alpha_array = zeros(error_rate_index,1);

for a = 1:error_rate_index
    alpha_array(a) = (a-1)*0.5;
end

%%
%plot error rates vs alpha
plot(alpha_array,error_rate,alpha_array,error_rate_train)

xlabel('alpha');
ylabel('Error rates (%)');
title('Q1: Plot of error rates vs. alpha'); 
legend('Test set', 'Training set','Location', 'southeast');

hold on;

plot(alpha_array(3),error_rate(3) ,'bx');
plot(alpha_array(21), error_rate(21),'bx');
plot(alpha_array(201), error_rate(201),'bx');

plot(alpha_array(3),error_rate_train(3),'rx');
plot(alpha_array(21), error_rate_train(21), 'rx');
plot(alpha_array(201), error_rate_train(201),'rx');

hold off;

%%
%show selected error rates
fprintf('Test error for alpha = 1 : %.4f\n',error_rate(3));
fprintf('Test error for alpha = 10 : %.4f\n',error_rate(21));
fprintf('Test error for alpha = 100 : %.4f\n',error_rate(201));

fprintf(' \n');

fprintf('Training error for alpha = 1 : %.4f\n',error_rate_train(3));
fprintf('Training error for alpha = 10 : %.4f\n',error_rate_train(21));
fprintf('Training error for alpha = 100 : %.4f\n',error_rate_train(201));
%%
