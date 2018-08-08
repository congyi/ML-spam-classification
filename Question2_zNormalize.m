clear global; clc

load('spamData.mat');

numFeatures = 57;

numTrain = length(Xtrain);
numTest = length(ytest);
numTotal =  numTrain + numTest;

%%pre-processing step
[Z_Xtest, Z_Xtrain] = PreProcess.zNormalize(Xtest, Xtrain);
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

zNorm_mu_1 = zeros(numFeatures,1);
zNorm_var_1 = zeros(numFeatures,1);
zNorm_mu_0 = zeros(numFeatures,1);
zNorm_var_0 = zeros(numFeatures,1);
%%

%%getting class conditional mean and variance for each feaure in class y=1
% p(xj(mu,var) | y=1);
for i = 1:numFeatures
    for j = 1:numTrain
            if ytrain(j) == 1
              zNorm_mu_1(i) = zNorm_mu_1(i) + Z_Xtrain(j,i);
            end 
    end      
    zNorm_mu_1(i) = zNorm_mu_1(i)./N1;  
        
end

for i = 1:numFeatures
    for j = 1:numTrain
            if ytrain(j) == 1
              zNorm_var_1(i) = zNorm_var_1(i) + (Z_Xtrain(j,i) - zNorm_mu_1(i))^2;
            end 
    end
  
    zNorm_var_1(i) = zNorm_var_1(i)./N1;     
end

%%getting class conditional mean and variance for each feaure in class y=0
% p(xj(mu,var) | y=0);
for i = 1:numFeatures
    for j = 1:numTrain
            if ytrain(j) == 0
              zNorm_mu_0(i) = zNorm_mu_0(i) + Z_Xtrain(j,i);
            end 
    end      
    zNorm_mu_0(i) = zNorm_mu_0(i)./N0;  
end

for i = 1:numFeatures
    for j = 1:numTrain
            if ytrain(j) == 0
              zNorm_var_0(i) = zNorm_var_0(i) + (Z_Xtrain(j,i) - zNorm_mu_0(i))^2;
            end 
    end
  
    zNorm_var_0(i) = zNorm_var_0(i)./N0;     
end

%%
%calculate log pi_c1 + log p(x_new|y=1) and log pi_c0 + log p(x_new|y=0)

logy1 = zeros(numTest,1);
logy0 = zeros(numTest,1);

for j = 1:numTest
    logy1(j) = log(pi_c1);
    for i = 1:numFeatures
        logy1(j) = logy1(j) + ...
            log(1/(sqrt(2*pi*zNorm_var_1(i))) * exp(-0.5*(Z_Xtest(j,i)-zNorm_mu_1(i))^2/zNorm_var_1(i)));
    end
end

%%prediction for test set
for j = 1:numTest
    logy0(j) = log(pi_c0);
    for i = 1:numFeatures
        logy0(j) = logy0(j) + ...
            log(1/sqrt((2*pi*zNorm_var_0(i))) * exp(-0.5*(Z_Xtest(j,i)-zNorm_mu_0(i))^2/zNorm_var_0(i)));
    end
end

spam_result = zeros(numTest,1);

for j = 1:numTest
    if logy1(j) > logy0(j)
        spam_result(j) = 1;
    elseif logy0(j) > logy1(j)
        spam_result(j) = 0;
    end
end

error_count = 0;

for j = 1:numTest
    if spam_result(j) ~= ytest(j)
       	error_count = error_count + 1;
    end
end

error_rate = error_count/numTest*100;
%%

%%prediction for training set
logy1_train = zeros(numTrain,1);
logy0_train = zeros(numTrain,1);

for j = 1:numTrain
    logy1(j) = log(pi_c1);
    for i = 1:numFeatures
        logy1(j) = logy1(j) + ...
            log(1/sqrt((2*pi*zNorm_var_1(i))) * exp(-0.5*(Z_Xtrain(j,i)-zNorm_mu_1(i))^2/zNorm_var_1(i)));
    end
end

for j = 1:numTrain
    logy0(j) = log(pi_c0);
    for i = 1:numFeatures
        logy0(j) = logy0(j) + ...
            log(1/(sqrt(2*pi*zNorm_var_0(i))) * exp(-0.5*(Z_Xtrain(j,i)-zNorm_mu_0(i))^2/zNorm_var_0(i)));
    end
end

spam_result_train = zeros(numTrain,1);

for j = 1:numTrain
    if logy1(j) > logy0(j)
        spam_result_train(j) = 1;
    elseif logy0(j) > logy1(j)
        spam_result_train(j) = 0;
    end
end

error_count_train = 0;

for j = 1:numTrain
    if spam_result_train(j) ~= ytrain(j)
       	error_count_train = error_count_train + 1;
    end
end

error_rate_train = error_count_train/numTrain*100;
%%

%show selected error rates
fprintf('z-Normalized data-set:\n');
disp_test = ['Error rate for test data set: ', num2str(error_rate), '%.'];
disp(disp_test);

disp_train = ['Error rate for training data set: ', num2str(error_rate_train), '%.'];
disp(disp_train);
%  

        

