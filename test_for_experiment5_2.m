%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
clear all
close all
load web-NotreDame
 G=Problem.A;
alpha=[0.85,0.90,0.95,0.99];                                               % damping factors to test
m=[3:10];                                                                    % restart steps to test
 tol=1e-10;
maxit=1000;
all_res=0;                                                                 % if all_res==1, the residual for every single iteration is recorded
digits(16);
mg= max(sum(G,1));                                                                          % else only the residual after every restart cycle is recorded
epsilon=1e-16;
c=1.01*(1+3.03*epsilon);
%% construct the vectors to store the output informations
it=zeros(length(alpha),length(m));
it_w=it;
solving_time=it;
solving_time_w=it;
mv=it;
mv_w=it;
res=cell(length(alpha),length(m));
res_w=cell(length(alpha),length(m));
res1=it;
res1_w=it;
res2=it;
res2_w=it;

for i=1:length(alpha)
    i;
 tol=2*epsilon*(3.03+alpha(i)*c*mg)/(1-epsilon*(3.03+alpha(i)*c*mg));
for j=1:length(m)
    j;
[x_w,it_w(i,j),solving_time_w(i,j),mv_w(i,j),res_w{i,j},res1_w(i,j),res2_w(i,j)]=weighted_FOM_pagerank_func(G,alpha(i),tol,m(j),maxit, all_res);
end
end
disp('web-NotreDame')
it_w
mv_w
 plot(m,solving_time_w(1,:),'-o',m,solving_time_w(2,:),'-o',m,solving_time_w(3,:),'-o',m,solving_time_w(4,:),'-o')