function [W,MSE] = trainingLinearClassifier(trainingSet, t, alpha, max_iterations, initalValue)
    [Ntot,dimx] = size(trainingSet);    
    [C,~] = size(t);
    W = ones(C,dimx+1);
    N = 500000;
    tol = 1e-10;

    if nargin == 5
        W = initalValue;
        N = max_iterations;
    elseif nargin == 4
        N = max_iterations;
    end
%     imagesPerClass = Ntot/C;
%     tk = eye(C);
%     tkAll = 
    n = 1;
    MSE_prev = inf;
    iterate = true;
    while iterate
        MSE_dW = zeros(size(W));
        MSE = 0;
        for k = 1:Ntot
            tk = t(:,k);
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
        n = n+1;
        diff = abs(MSE - MSE_prev);
        MSE_prev = MSE;
        iterate = diff > tol && n <= N;
    end
    diff
    MSE
    n
end

