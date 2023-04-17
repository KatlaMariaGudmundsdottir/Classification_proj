function g = sigmoid(z)
    [nz,Ntot] = size(z);
    g = zeros(size(z));
    for i = 1:Ntot
        for j = 1:nz
            g(j,i)  = 1/(1 + exp(-z(j,i)));
        end
    end
end

