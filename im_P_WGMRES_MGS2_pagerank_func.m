%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
%% input: s: the degree of polynomial in the polynomial preconditioner
%% input: gamma: the shift parameter to construct the non-singular system for building the preconditioner 
function [it,solving_time,mv,res,res1,res2]=im_P_WGMRES_MGS2_pagerank_func(G,alpha,tol,m,maxit,s,gamma,all_res)
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
P1=(alpha/(1+gamma))*P;
P=I-alpha*P;
h=-per;
f=alpha*index1+(1-alpha)*ones(1,n);
h1=-(1/(1+gamma))*h;
%--------------------------------------------

%------parameters for weighted arnoldi-------------------------
wt=ones(n,1); % weight
x0=per;
e=zeros(m+1,1);
e(1)=1;
mv=0;

%----start the restart weighted FOM----------------------------
tic
r0=im_poly_prec(P1,h1,f,gamma,s,x0);
mv=mv+s;
r0=-mv_pr(r0);
mv=mv+1;
res=norm(r0,2)*ones(1,s+1);
it=1;
V=zeros(n,m);
H=zeros(m+1,m);

while it<maxit 
    mv_pre=mv;
beta=normw(r0,wt);
V(:,1)=r0/beta;

for j=1:m
temV(j).v=V(:,j);
wttemV(j).v=wt.*temV(j).v;
w=im_poly_prec(P1,h1,f,gamma,s,temV(j).v);
mv=mv+s;
    w=mv_pr(w);   
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
 temHmy=temHm*y;
 r0=r0-[V,vm1]*(temHmy);
mv=mv+1;
k=1/(1+gamma);
ka=k;
norm_x=norm(x0,1)*k*(1-ka^(s+1))/(1-ka);
now_res=norm(r0,2)/norm_x;
if all_res==1
res=[res,res(end)*ones(1,mv-mv_pre-1),now_res];
else
   res=[res,now_res];
end 
if (res(end))<tol %res(1)
    break;
end
it=it+1;
 tem=(abs(r0));
 wt=tem/norm(tem,1);
end

x=im_poly_prec(P1,h1,f,gamma,s,x0);
mv=mv+s;
solving_time=toc;
res=[res,res(end)*ones(1,s)];
res1=res(end);
res2=norm(-mv_pr(x),2)/norm(x,1);

 