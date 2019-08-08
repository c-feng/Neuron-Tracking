function prob=ProbvecSimilar(N1,N2,Sigma)
currs=N1'*N2;
prob=Sigma*sqrt(pi/2)*(1+erf(currs/(sqrt(2)*Sigma)));
prob=prob*exp(-(1-currs^2)/(2*Sigma^2));






