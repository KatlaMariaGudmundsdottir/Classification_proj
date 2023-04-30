clear
x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');


W = [0.222405612687051	2.45617908160929	-3.37821387881523	-0.932779985742882	1.16916142012784;
    0.891591036120648	-2.31150906518052	0.0851130237137354	-1.14317469922865	2.20035057838884;
    -3.52338253866417	-2.91303866315534	4.97226783685800	4.41704620945179	-1.41944945445024]

xtestC = [x3all(36,:)';1];
zk = W*xtestC
gk = sigmoid(zk)


function gk = sigmoid(zk)
    nz = length(zk);
    gk = zeros(nz, 1);
    for i = 1:nz
        gk(i)  = 1/(1 + exp(-zk(i)));
    end
end