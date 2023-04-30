function [g, confmat] = linearClassifier(x,t,W)
    [Ntot,dimx] = size(x); 
    x = [x'; ones(1,Ntot)];
    z = W*x;
    g = sigmoid(z);
    res = zeros(1,Ntot);
    trueClass = zeros(1,Ntot);
    for i = 1:Ntot 
        res(1,i) = find(g(:,i)==max(g(:,i)));
        trueClass(1,i) = find(t(:,i)==max(t(:,i)));
    end
    confmat = confusionmat(trueClass,res);
end