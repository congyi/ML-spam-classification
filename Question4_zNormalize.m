clear global; clc

load('spamData.mat');

%%
%initializing variables
numFeatures = 57;
numTrain = length(Xtrain);
numTest = length(ytest);
numTotal =  numTrain + numTest;

k_array = [1:10,15:5:100];

error_count = zeros(length(k_array), 1);
error_rate = zeros(length(k_array), 1);
error_count_train = zeros(length(k_array), 1);
error_rate_train = zeros(length(k_array), 1);

y1_count = zeros(numTest, length(k_array));
y1_count_train = zeros(numTrain, length(k_array));

spam_result = zeros(numTest, length(k_array));
spam_result_train = zeros(numTrain, length(k_array));
    
dist_matrix = zeros(numTest,numTrain);
dist_matrix_train = zeros(numTrain,numTrain);

%%
%pre-processing step
[Z_Xtest, Z_Xtrain] = PreProcess.zNormalize(Xtest, Xtrain);

%%
%%distance computation for test set
for i=1:numTest       % for each data-line in test set...    
    for n=1:numTrain  %calculate eclu. dst to each data-line in training set
        dist_matrix(i,n) = norm(Z_Xtest(i,:) - Z_Xtrain(n,:));      
    end
end   
    
%%
%distance computation for training set
for i=1:numTrain       % for each data-line in training set...    
  for n=1:numTrain  %calculate eclu. dst to each data-line in training set
        dist_matrix_train(i,n) = norm(Z_Xtrain(i,:) - Z_Xtrain(n,:));   
  end
end

%%
%MAIN LOOP STARTS HERE
for k=1:length(k_array)

    %computing k-NN and error rates for test set
    for i=1:numTest
        
        to_sort = horzcat(ytrain,transpose(dist_matrix(i,:))); %%tag on labels
        sorted = sortrows(to_sort,2); %sort by column 2 - the distance

        %of the k-nearest neighbours, check how many are 1's
        for cnt = 1:k_array(k)
            if sorted(cnt,1) == 1
                y1_count(i,k) = y1_count(i,k) + 1;
            end
        end

        %if more than half of k's are 1's, more likely to be spam
        if (y1_count(i,k)/k_array(k)) > 0.5
            spam_result(i,k) = 1;
        end 

       if spam_result(i,k) ~= ytest(i)
            error_count(k) = error_count(k) + 1;
       end
      
    end
    
    error_rate(k) = error_count(k)/numTest*100;
    
    %computing k-NN and error rates training set
    for i=1:numTrain
        
        to_sort_train = horzcat(ytrain,transpose(dist_matrix_train(i,:))); %%tag on labels
        sorted_train = sortrows(to_sort_train,2); %sort by column 2 - the distance

        %of the k-nearest neighbours, check how many are 1's
        for cnt = 1:k_array(k)
            if sorted_train(cnt,1) == 1
               y1_count_train(i,k) = y1_count_train(i,k) + 1;
            end
        end

       %if more than half of k's are 1's, more likely to be spam
       if (y1_count_train(i,k)/k_array(k)) > 0.5
           spam_result_train(i,k) = 1;
       end 

       if spam_result_train(i,k) ~= ytrain(i)
           error_count_train(k) = error_count_train(k) + 1;
       end
       
    end
    
    error_rate_train(k) = error_count_train(k)/numTrain*100;
    
end
%%
%plotting the results
plot(k_array,error_rate,k_array,error_rate_train)

xlabel('k');
ylabel('Error rates (%)');
title('Q4(z-Normalized): Plot of error rates vs. k'); 
legend('Test set', 'Training set','Location', 'southeast');

hold on;

plot(k_array(1),error_rate(1) ,'bx');
plot(k_array(10), error_rate(10),'bx');
plot(k_array(28), error_rate(28),'bx');

plot(k_array(1),error_rate_train(1),'rx');
plot(k_array(10), error_rate_train(10), 'rx');
plot(k_array(28), error_rate_train(28),'rx');

hold off;

%%
%show selected error rates
fprintf('z-Normalized data-set:\n');
fprintf('Test error for k = 1 : %.4f\n',error_rate(1));
fprintf('Test error for k = 10 : %.4f\n',error_rate(10));
fprintf('Test error for k = 100 : %.4f\n',error_rate(28));

fprintf(' \n');

fprintf('Training error for k = 1 : %.4f\n',error_rate_train(1));
fprintf('Training error for k = 10 : %.4f\n',error_rate_train(10));
fprintf('Training error for k = 100 : %.4f\n',error_rate_train(28));