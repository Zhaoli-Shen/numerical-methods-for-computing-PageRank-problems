function [mv,x,solving_time,res] = MPIO_new(G,alpha,beta,tol,eta,m,all_res)
%INNER_OUTER 此处显示有关此函数的摘要
%   此处显示详细说明
weighted_FOM_global
%---------------generate pagerank problem------------------------
n=size(G,2);
I=speye(n);
v=1/n*ones(n,1);


sumrow=full(sum(G,2));
dangling=find(sumrow==0);
d=zeros(1,n);
d(dangling)=1;
sumrow(dangling)=1;
P=diag(sparse(1./sumrow))*G;
original_P=P;
clear sumrow
clear dangling
clear G;
% P=P';
P=I-alpha*P;
h=-v;
f=alpha*d+(1-alpha)*ones(1,n);
%-----------------------------------------
x0=v;
x_initial=x0;

%-----------------------------------------
mv=0;
x=x_initial;
tic
res(1)=norm(mv_pr(x),2);
mv=mv+1;
while res(end)>=tol
    mv_pre=mv;
for i=1:m       
    x=x-mv_pr(x); 
    mv=mv+1;
end

y=(original_P'*x+(d*x)*v);
f_new=(alpha-beta)*y+(1-alpha)*v;
mv=mv+1;
while 1
x=beta*y+f_new;
y=(original_P'*x+(d*x)*v);
mv=mv+1;
if norm(f_new+beta*y-x,2)<eta
break;
end
end
x_new=alpha*y+(1-alpha)*v;
if all_res==1
  res=[res,res(end)*ones(1,mv-mv_pre-1),norm(x_new-x,2)];
else
    res=[res,norm(x_new-x,2)];
end
end
solving_time=toc;
end

