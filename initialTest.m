x1all = load('class_1','-ascii');

x1test = [x1all(1:30,:)];
[Ntot,dimx] = size(x1test);
 
alpha = 0.1;
 
W = zeros(dimx,Ntot*dimx);
W(1:end,1:dimx) = eye(dimx);
    
t = 0.33;

for k = 1:Ntot
   
end

for k = 1:Ntot
    W(1:end,dimx*k:dimx*k+1) = W(1:end,dimx*k-1:dimx*k);
end 
