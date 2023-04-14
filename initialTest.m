x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');

x1test = [x1all(1:30,:)];
x2test = [x2all(1:30,:)];
x3test = [x3all(1:30,:)];
xtest = [x1test; x2test; x3test];

C = 3;

[Ntot,dimx] = size(xtest);
 
alpha = 0.01;
 
W = ones(C,dimx+1);

tk1 = 0.1666;
tk2 = 0.5;
tk3 = 0.833;
%tk = [tk1; tk2; tk3];

for m = 1:3000
    MSE_dW = zeros(size(W));
    MSE = 0;
    for k = 1:90
        switch k
            case k<=30
                tk = [1;0;0];
            case (31 <= k) & (k <= 60)
                tk = [0;1;0];
            case 60 <= k
                tk = [0; 0;1];
        end
        xk = [xtest(k,:)'; 1];
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

function gk = sigmoid(zk)
    nz = length(zk);
    gk = zeros(nz, 1);
    for i = 1:nz
        gk(i)  = 1/(1 + exp(-zk(i)));
    end
end
    