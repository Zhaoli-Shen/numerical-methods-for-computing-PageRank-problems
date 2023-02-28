%% polynomial preconditioning in the PW-FOM for PageRank computation, solving (I-A)x=0 
%% written by Z.-L. Shen, UESTC
%% related paper: preconditioned weighted FOM for PageRank computations (under review)
function y=im_poly_prec(P,h,f,gamma,s,x)
y=x;
for i=1:s
    y=(P'*y+h*(f*y))+x;
end

y=y/(1+gamma);
end