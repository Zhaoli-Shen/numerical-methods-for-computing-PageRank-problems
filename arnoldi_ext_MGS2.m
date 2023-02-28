function [it,solving_time,mv,res,res1,res2] = arnoldi_ext_MGS2(G,alpha,tol,m,all_res)
%ARNOLDI_EXT 此处显示有关此函数的摘要
%   此处显示详细说明
%% combination of Arnoldi method and Power-extrapolation method 
%% written by Z.-L. Shen, SAU
%% related paper: A new extrapolation method for PageRank computations
%% convergence criterion ||Ax-x ||_2/||(1-alpha)v ||_2<tol, A=alpha*(P+v*d')+(1-alpha)*v*e';
%%        initialization of the PagerRank problem 
%%        fomating A into A=P'+h*f                                                    % damping factor for pagerank problem
n=size(G,2);                                                    % dimension of the web adjacency matrix G
v=1/n*ones(n,1);                                          % the personalization vector
sumrow=full(sum(G,2));
dangling=find(sumrow==0);                     % set of dangling nodes
l=length(dangling);                                      % amount of dangling nodes
mu=(1-alpha)+alpha*l/n;  
d=zeros(1,n);
d(dangling)=1;                                               %  the index vector representing dangling nodes 
sumrow(dangling)=1;
P=diag(sparse(1./sumrow))*G;               % the transition matrix of the web link graph
clear subrow;
clear dangling;
clear G;
%%%%%%% fomating A into A=P'+h*f %%%%%%%%%%%%%%%%
P=alpha*P;                           
h=v;            
f=alpha*d+(1-alpha)*ones(1,n);  


%% parameters settings
 tol_power=1e-6;                                           % tol for the pow-extrpolation method                                                                               % dimension of the krylov subspace of arnoldi
m_power=40 ;                                                 % window size for the extraplotion operation
maxit=10000;                                                 % limit of the amount of iterations
kk=ones(1,m-1);
x=v;                                                                   % initial guess





%% start the power-extrapolation method

normr=norm(P'*x+(f*x)*h-x,2);   
tic  
res(1)=normr;
mv=0;
it=0;  
                                                        
while(normr>tol_power && mv<20000)
     x_tem=P'*x;
     x_tem=x_tem+(f*x)*h;
     mv=mv+1;
     normr=norm(x_tem-x,2);
     res(mv)=normr;
   if normr<tol_power
         break;
   end
 if(mod(it,m_power)==0)
   x_pre=x_tem; 
end
if(mod(it,m_power)==1&& (it~=1))   
   % ---extrapolation
   x_tem=x_tem-(mu-1)*x_pre;
   x_tem=x_tem/norm(x_tem,1);
end
x=x_tem;
it=it+1;    
end

clear x_tem;
clear x_pre

%%   start the restart arnoldi
V=zeros(n,m);
H=zeros(m+1,m);
MI=zeros(m+1,m);
MI(1:m,1:m)=eye(m);

x0=x;                                                                 %using the soultion of Power-extrapolation as the initial guess 
clear x
it=0;
while mv<=20000 
V(:,1)=x0/norm(x0,2);
for j=1:m  
temV(j).v=V(:,j);    
w=P'*temV(j).v;
w=w+(f*temV(j).v)*h;
mv=mv+1;
    
    for k=1:2
    if k==1
    for i=1:j
        temVi=temV(i).v;
        temhij=(w')*(temVi);
        H(i,j)=temhij; % *****important
        w=w-temhij*temVi;
    end
    else
       for i=1:j
        temhij=(w')*(temVi);
        H(i,j)=H(i,j)+temhij; % *****important
        w=w-temhij*temVi;
       end
    end
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
[U,S,W]=svd(H-MI);
x0=V*(W(:,m));
%----------------------------------------------
norm_x0=norm(x0,1);
if all_res==1
res=[res,res(end)*kk,(S(m,m)/norm_x0)];
else
    res=[res,(S(m,m)/norm_x0)];
end
it=it+1;
if (res(end))<tol
    break;
end
end
x0=x0/norm_x0;                                  %normalization of the solution
solving_time=toc;
res1=res(end);
res2=norm(P'*x0+(f*x0)*h-x0,2);

end

