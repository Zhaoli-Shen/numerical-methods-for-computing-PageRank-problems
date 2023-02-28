%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
function [it,solving_time,mv,res,res1,res2]=weighted_GMRES_MGS2_pagerank_func(G,alpha,tol,m,maxit,all_res)
weighted_FOM_global
%-------------------------generate pagerank problems-----------
n=size(G,2); %读取邻接矩阵G的维数
I=speye(n);%建立n维对角矩阵
per=1/n*ones(n,1);%建立个性化向量


sumrow=full(sum(G,2));%读取邻接矩阵G的行和
dangling=find(sumrow==0);%找到矩阵G中行和为0的点，对应于dangling nodes
index1=zeros(1,n);%建立index矩阵，初始化为一行n列的零向量
index1(dangling)=1;%将index1中
sumrow(dangling)=1;
P=diag(sparse(1./sumrow))*G;
clear subrow;
clear dangling;
clear G;
P=I-alpha*P;
h=-per;
f=alpha*index1+(1-alpha)*ones(1,n);
%-------------------- (I-A)*x=P'*x+h*(f*x);

%------parameters for weighted arnoldi-------------------------
wt=ones(n,1); % weight
x0=per;
e=zeros(m+1,1);
e(1)=1;
mv=0;
kk=ones(1,m);
%----start the restart weighted FOM----------------------------
tic;
r0=-mv_pr(x0);
res=norm(r0,2);
mv=mv+1;
it=1;
V=zeros(n,m);
H=zeros(m+1,m);
while it<maxit 
beta=normw(r0,wt);
V(:,1)=r0/beta;

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
temHm=[H(1:m,:);zeros(1,m)];
temHm(m+1,m)=hlast;
betae=beta*e;
y=temHm\betae;
x0=x0+V*y;
r0=-mv_pr(x0);
mv=mv+1;
now_res=norm(r0,2)/norm(x0,1);
if all_res==1
res=[res,res(end)*kk,now_res];
else
   res=[res,now_res];
end 
if (res(end))<tol
    break;
end
it=it+1;
%----------------------------------
% 
 tem=(abs(r0))/norm(r0,1);
 wt=tem;
end
solving_time=toc;
res1=res(end);
res2=norm(-mv_pr(x0),2)/norm(x0,1);


