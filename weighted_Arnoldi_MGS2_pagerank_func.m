%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded 
function [it,solving_time,mv,res,res1,res2]=weighted_Arnoldi_MGS2_pagerank_func(G,alpha,tol,m,maxit,all_res)
weighted_FOM_global
%---------------generate pagerank problems------------------------------
n=size(G,2);
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
P=alpha*P;
h=per;
f=alpha*index1+(1-alpha)*ones(1,n);
%-------------------- A*x=P'*x+h*(f*x)

wt=ones(n,1); % weight
x0=per;
mv=0;
it=1;
V=zeros(n,m);
H=zeros(m+1,m);
MI=zeros(m+1,m);
MI(1:m,1:m)=eye(m);
kk=ones(1,m-1);
%----start the restart weighted arnoldi----------------------------
tic;
 r0=x0-mv_pr(x0);
 mv=mv+1;
 res=norm(r0,2);


while mv<=20000 
V(:,1)=x0/normw(x0,wt);
for j=1:m
temV(j).v=V(:,j);
wttemV(j).v=wt.*temV(j).v;
    w=mv_pr(temV(j).v); 
    mv=mv+1;
    for k=1:2
    if k==1
    for i=1:j
        temVi=temV(i).v;
        temhij=(w')*wttemV(i).v;
        H(i,j)=temhij; % *****important
        w=w-temhij*temVi;
    end
    else
       for i=1:j
        temhij=(w')*wttemV(i).v;
        H(i,j)=H(i,j)+temhij; % *****important
        w=w-temhij*temVi;
       end
    end
    end
    hlast=normw(w,wt);
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
[U,S,W]=svd(H-MI);
um=U(:,m);
deta=diag(S);
deta=deta(end);
rm=deta*(V*um(1:m)+um(end)*vm1);
x0=V*(W(:,m));
now_res=norm(rm,2)/norm(x0,1);
if all_res==1
res=[res,res(end)*kk,now_res];
else
   res=[res,now_res];
end 
if (res(end))<tol
    break;
end
it=it+1;

 tem=(abs(rm));
 wt=tem/norm(tem,1);
end
solving_time=toc;
res1=res(end);
res2=norm(x0-mv_pr(x0),2)/norm(x0,1);
