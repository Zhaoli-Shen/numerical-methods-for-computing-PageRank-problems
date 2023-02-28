%% proposed by H.-F. Zhang, UESTC
%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
function [it,solving_time,mv,res,res1,res2]=GMRES_pagerank_func(G,alpha,tol,m,maxit,all_res)
weighted_FOM_global
%-------------------------generate pagerank problems-----------
n=size(G,2);
I=speye(n);
per=1/n*ones(n,1);
sumrow=full(sum(G,2));
dangling=find(sumrow==0);
index1=zeros(1,n);
index1(dangling)=1;
sumrow(dangling)=1;
P=diag(sparse(1./sumrow))*G;
clear subrow;
clear dangling;
clear G;
P=I-alpha*P;
h=-per;
f=alpha*index1+(1-alpha)*ones(1,n);
%-------------------- (I-A)*x=P'*x+h*(f*x);

%------parameters for GMRES-------------------------
x0=per;
e=zeros(m+1,1);
e(1)=1;
mv=0;
kk=ones(1,m);
%----start the restart weighted FOM----------------------------
tic;
r0=-mv_pr(x0);
mv=mv+1;
res=norm(r0,2);

it=1;
V=zeros(n,m);
H=zeros(m+1,m);
while it<maxit 
beta=norm(r0,2);
V(:,1)=r0/beta;
for j=1:m
temV(j).v=V(:,j);
    w=mv_pr(temV(j).v);  
    mv=mv+1;
    for i=1:j
        temVi=temV(i).v;
        temhij=(w')*(temVi);
        H(i,j)=temhij; % *****important
        w=w-temhij*temVi;
    end
    hlast=norm(w,2);
    H(j+1,j)=hlast;   % *****important
    if hlast==0
         disp('break down');
        break;       
    end
    vm1=(1/hlast)*w;
    if j<m
    V(:,j+1)=vm1;
    end
    
end
temHm=[H(1:m,:);zeros(1,m)];
temHm(m+1,m)=hlast;
betae=beta*e;
y=temHm\betae;
x0=x0+V*y;
r0=-mv_pr(x0);
mv=mv+1;
now_res=norm(r0)/norm(x0,1);
if all_res==1
res=[res,res(end)*kk,now_res];
else
res=[res,now_res];
end   
if (res(end))<=tol
    break;
end
it=it+1;
end
solving_time=toc;
it;
mv;
res1=res(end);
res2=norm(-mv_pr(x0),2)/norm(x0,1);
