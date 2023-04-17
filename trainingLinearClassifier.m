function [W,MSE] = trainingLinearClassifier(trainingSet, numberOfClasses, alpha, iterations, varargin)
    [Ntot,dimx] = size(trainingSet);    
    C = numberOfClasses;
    if nargin == 5
        W = varargin{1};
    else 
        W = ones(C,dimx+1);
    end
%     imagesPerClass = Ntot/C;
%     tk = eye(C);
%     tkAll = 
    for m = 1:iterations
        MSE_dW = zeros(size(W));
        MSE = 0;
        for k = 1:Ntot
            if (k < 31) 
                tk = [1;0;0];
            elseif (k > 30 && k < 61)
                tk = [0;1;0];
            elseif (k > 60 && k < 91)
                tk = [0;0;1];
            end
            xk = [trainingSet(k,:)'; 1];
            zk = W*xk;
            gk = sigmoid(zk);
            MSE_dgk = gk-tk;
            g_dzk = dot(gk, (1-gk));
            zk_dW = xk';
            MSE_dW = MSE_dW + MSE_dgk*g_dzk*zk_dW;
            MSE = MSE + 1/2*((gk-tk)')*(gk-tk);
        end 
        W = W - alpha*MSE_dW;
        disp(MSE)
    end
end

