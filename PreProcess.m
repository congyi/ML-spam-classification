classdef PreProcess
   
    methods(Static)
        function [P_Xtest,P_Xtrain] = binariZe(Xtest, Xtrain)
            P_Xtest = double(logical(Xtest));
            P_Xtrain = double(logical(Xtrain));
        end
        
        function [P_Xtest,P_Xtrain] = logTransform(Xtest, Xtrain)
            P_Xtest = log(Xtest + 0.1);
            P_Xtrain = log(Xtrain + 0.1);
        end
         
        function [P_Xtest,P_Xtrain] = zNormalize(Xtest, Xtrain)
            P_Xtest = zscore(Xtest);
            P_Xtrain = zscore(Xtrain);
         end
    end  
end