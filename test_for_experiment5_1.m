%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)
%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
clear all
load web-Stanford;
 G=Problem.A;
alpha=[0.99];                                               % damping factors to test
m=[3,6,9,15,20];                                                                    % restart steps to test
maxit=1000;
all_res=0;                                                                 % if all_res==1, the residual for every single iteration is recorded
digits(16);
mg= max(sum(G,1));                                                                          % else only the residual after every restart cycle is recorded
epsilon=1e-16;
c=1.01*(1+3.03*epsilon);
%% construct the vectors to store the output informations
it=zeros(length(alpha),length(m));
it_w=it;
it_wr=it;
solving_time=it;
solving_time_w=it;
solving_time_wr=it;
mv=it;
mv_w=it;
mv_wr=it;
res=cell(length(alpha),length(m));
res_w=cell(length(alpha),length(m));
res_wr=cell(length(alpha),length(m));
res1=it;
res1_w=it;
res1_wr=it;
res2=it;
res2_w=it;
res2_wr=it;
dis=-1*ones(length(alpha),length(m));
for i=1:length(alpha)
 tol=min(1e-10,2*epsilon*(3.03+alpha(i)*c*mg)/(1-epsilon*(3.03+alpha(i)*c*mg)))
for j=1:length(m)
[x_w,it_w(i,j),solving_time_w(i,j),mv_w(i,j),res_w{i,j},res1_w(i,j),res2_w(i,j)]=FOM_pagerank_func(G,alpha(i),tol,m(j),maxit, all_res);
[x_wr,it_wr(i,j),solving_time_wr(i,j),mv_wr(i,j),res_wr{i,j},res1_wr(i,j),res2_wr(i,j)]=FOM_realr_pagerank_func(G,alpha(i),tol,m(j),maxit, all_res);
[x_w_gs2,it_gs2(i,j),solving_time_gs2(i,j),mv_gs2(i,j),res_gs2{i,j},res1_gs2(i,j),res2_gs2(i,j)]=FOM_MGS2_pagerank_func(G,alpha(i),tol,m(j),maxit, all_res);
[x_gs2r,it_gs2r(i,j),solving_time_gs2r(i,j),mv_gs2r(i,j),res_gs2r{i,j},res1_gs2r(i,j),res2_gs2r(i,j)]=FOM_MGS2_realr_pagerank_func(G,alpha(i),tol,m(j),maxit, all_res);
end
end
disp('cnr-2000')
it_w
it_wr
it_gs2
it_gs2r
solving_time_w
solving_time_wr
solving_time_gs2
solving_time_gs2r