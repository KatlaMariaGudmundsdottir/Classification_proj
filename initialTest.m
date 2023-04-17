x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');

x1training = [x1all(1:30,:)];
tk1training = kron([1;0;0],ones(1,30));
x2training = [x2all(1:30,:)];
tk2training = kron([0;1;0],ones(1,30));
x3training = [x3all(1:30,:)];
tk3training = kron([0;0;1],ones(1,30));
xtraining = [x1training; x2training; x3training];
tktraining = [tk1training,tk2training,tk3training];

C = 3;

[Ntot,dimx] = size(xtest);
 
alpha = 0.005;
 
[W,MSE] = trainingLinearClassifier(xtest, C, alpha, 10000)
[g,confmat] = linearClassifier(xtraining, tktraining, W)