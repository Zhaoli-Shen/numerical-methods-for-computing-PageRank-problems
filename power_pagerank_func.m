%% Power method for PageRank computation, solving Ax=x 
%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)
%% convergence criterion ||Ax-x ||_2<tol, A=alpha*(P+v*d')+(1-alpha)*v*e';


%% output: it: the number of iterations
function [it,solving_time,mv,res,x0]=power_pagerank_func(G,alpha,tol,maxit)
weighted_FOM_global
%---------------generate pagerank problem------------------------
n=size(G,2);
I=speye(n);
per=1/n*ones(n,1);
sumrow=full(sum(G,2));
dangling=find(sumrow==0);
index1=zeros(1,n);
index1(dangling)=1;
sumrow(dangling)=1;
P=diag(sparse(1./sumrow))*G;
clear sumrow
clear dangling
clear G;
P=I-alpha*P;
h=-per;
f=alpha*index1+(1-alpha)*ones(1,n);
%-----------------------------------------
x0=per;
x_initial=x0;

%-----------------------------------------
mv=0;
x0=x_initial;
tic
r0=-mv_pr(x0);
mv=mv+1;
res(1)=norm(r0,2);


for i=1:maxit    
    x0=r0+x0;
    r0=-mv_pr(x0);
    mv=mv+1;
    res(i+1)=norm(r0,2);
    if res(end)<tol
        break;
    end
    
end
solving_time=toc;
it=i;