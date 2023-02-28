%% test several methods for PageRank computation, solving (I-A)x=0 
%% written by Z.-L. Shen, UESTC
%% related paper: preconditioned weighted FOM for PageRank computations (under review)
clc
clear all
close all
%% load the problem
load web-Stanford;
% or
% load xx.mat
  G=Problem.A;
%% set the parameters
alpha=[0.85,0.90,0.95,0.99];                                               % damping factors to test
m=8;                                                                       % restart steps
tol=1e-10;                                                                 
maxit=2500;
gamma=0.1;                                                  % shift parameter to construct non-singular system for building preconditioner
all_res=1;                                                  % if all_res==1, the residual for every single iteration is recorded                                                            % else only the residual after every restart cycle is recorded
l=[2,3,4,5];                                                % the degree of the polynomial in the preconditioner corresponding to each alpha 
write_or_not=1;
%% construct the vectors to store the output informations
it=zeros(length(alpha),1);
it_mgs2=it;
it_g=it;
it_w=it;
it_wg=it;
it_p=zeros(length(alpha),1);
it_pg=zeros(length(alpha),1);
it_p_uw=zeros(length(alpha),1);
it_a=it;
it_ae=it;
it_power=it;
solving_time=it;
solving_time_mgs2=it;
solving_time_g=it;
solving_time_w=it;
solving_time_wg=it;
solving_time_p=zeros(length(alpha),1);
solving_time_pg=zeros(length(alpha),1);
solving_time_p_uw=zeros(length(alpha),1);
solving_time_a=it;
solving_time_ae=it;
solving_time_power=it;
solving_time_m=it;
mv=it;
mv_mgs2=it;
mv_g=it;
mv_w=it;
mv_wg=it;
mv_p=zeros(length(alpha),1);
mv_pg=zeros(length(alpha),1);
mv_p_uw=zeros(length(alpha),1);
mv_a=it;
mv_ae=it;
mv_power=it;
mv_m=it;
res=cell(length(alpha),1);
res_mgs2=res;
res_g=cell(length(alpha),1);
res_w=cell(length(alpha),1);
res_wg=cell(length(alpha),1);
res_p=cell(length(alpha),1);
res_pg=cell(length(alpha),1);
res_p_uw=cell(length(alpha),1);
res_a=cell(length(alpha),1);
res_ae=cell(length(alpha),1);
res_power=cell(length(alpha),1);
res_m=cell(length(alpha),1);
res1=it;
res1_mgs2=it;
res1_g=it;
res1_w=it;
res1_wg=it;
res1_p=zeros(length(alpha),1);
res1_pg=zeros(length(alpha),1);
res1_p_uw=zeros(length(alpha),1);
res1_a=it;
res1_ae=zeros(length(alpha),1);
res2_ae=res1_ae;
res2=it;
res2_mgs2=it;
res2_g=it;
res2_w=it;
res2_wg=it;
res2_p=zeros(length(alpha),1);
res2_pg=zeros(length(alpha),1);
res2_p_uw=zeros(length(alpha),1);
res2_a=it;
% set the convergence tolerence 
digits(16);
mg= max(sum(G,1));                                                                          % else only the residual after every restart cycle is recorded
epsilon=1e-16;
c=1.01*(1+3.03*epsilon);

%% run the methods 
%% remark that the first output res, e.g. res{i} is a vector contains the residual of each iteration
%% the last two residuals are the approximate residual and the real residual after the last iteration
for i=1:length(alpha)
 alpha(i)
 tol=2*epsilon*(3.03+alpha(i)*c*mg)/(1-epsilon*(3.03+alpha(i)*c*mg))
 disp('FOM')
  [x,it(i),solving_time(i),mv(i),res{i},res1(i),res2(i)]=FOM_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
   disp('FOM-mgs2')
  [x,it_mgs2(i),solving_time_mgs2(i),mv_mgs2(i),res_mgs2{i},res1_mgs2(i),res2_mgs2(i)]=FOM_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
   disp('GMRES')
  [it_g(i),solving_time_g(i),mv_g(i),res_g{i},res1_g(i),res2_g(i)]=GMRES_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
   disp('GMRES-mgs2')
  [it_g_mgs2(i),solving_time_g_mgs2(i),mv_g_mgs2(i),res_g_mgs2{i},res1_g_mgs2(i),res2_g_mgs2(i)]=GMRES_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
  disp('wFOM')
 [x,it_w(i),solving_time_w(i),mv_w(i),res_w{i},res1_w(i),res2_w(i)]=weighted_FOM_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
  disp('wFOM-mgs2')
 [x,it_w_mgs2(i),solving_time_w_mgs2(i),mv_w_mgs2(i),res_w_mgs2{i},res1_w_mgs2(i),res2_w_mgs2(i)]=weighted_FOM_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
 disp('wGMRES')
 [it_wg(i),solving_time_wg(i),mv_wg(i),res_wg{i},res1_wg(i),res2_wg(i)]=weighted_GMRES_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
  disp('wGMRES-mgs2')
 [it_wg_mgs2(i),solving_time_wg_mgs2(i),mv_wg_mgs2(i),res_wg_mgs2{i},res1_wg_mgs2(i),res2_wg_mgs2(i)]=weighted_GMRES_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
disp('Pw-FOM')
[it_p(i),solving_time_p(i),mv_p(i),res_p{i},res1_p(i),res2_p(i)]=im_P_WFOM_pagerank_func(G,alpha(i),tol,m,maxit,l(i),gamma,all_res);
disp('Pw-FOM-mgs2')
[it_p_mgs2(i),solving_time_p_mgs2(i),mv_p_mgs2(i),res_p_mgs2{i},res1_p_mgs2(i),res2_p_mgs2(i)]=im_P_WFOM_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,l(i),gamma,all_res);
disp('Pw-GMRES')
[it_pg(i),solving_time_pg(i),mv_pg(i),res_pg{i},res1_pg(i),res2_pg(i)]=im_P_WGMRES_pagerank_func(G,alpha(i),tol,m,maxit,l(i),gamma,all_res);
disp('Pw-GMRES-mgs2')
[it_pg_mgs2(i),solving_time_pg_mgs2(i),mv_pg_mgs2(i),res_pg_mgs2{i},res1_pg_mgs2(i),res2_pg_mgs2(i)]=im_P_WGMRES_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,l(i),gamma,all_res);
disp('w-Arnoldi')
 [it_a(i),solving_time_a(i),mv_a(i),res_a{i},res1_a(i),res2_a(i)]=weighted_Arnoldi_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
 disp('w-Arnoldi-mgs2')
 [it_a_mgs2(i),solving_time_a_mgs2(i),mv_a_mgs2(i),res_a_mgs2{i},res1_a_mgs2(i),res2_a_mgs2(i)]=weighted_Arnoldi_MGS2_pagerank_func(G,alpha(i),tol,m,maxit,all_res);
 disp('Power')
 [it_power(i),solving_time_power(i),mv_power(i),res_power{i},x_power]=power_pagerank_func(G,alpha(i),tol,20000);
  disp('Arnoldi-ext')
[it_ae(i),solving_time_ae(i),mv_ae(i),res_ae{i},res1_ae(i),res2_ae(i)] = arnoldi_ext(G,alpha(i),tol,m,all_res);
disp('Arnoldi-ext-MGS2')
[it_ae_mgs2(i),solving_time_ae_mgs2(i),mv_ae_mgs2(i),res_ae_mgs2{i},res1_ae_mgs2(i),res2_ae_mgs2(i)] = arnoldi_ext_MGS2(G,alpha(i),tol,m,all_res);
  disp('MPIO_new')
[mv_m(i),x_m,solving_time_m(i),res_m{i}] = MPIO_new(G,alpha(i),0.5,tol,1e-4,7,all_res);
figure
semilogy(1:mv(i),res{i},'-o',1:mv_g(i),res_g{i},'-d',1:mv_w(i),res_w{i},'-+',1:mv_wg(i),res_wg{i},'-^',1:mv_p(i),res_p{i},'-*',1:mv_pg(i),res_pg{i},'-v',1:mv_a(i),res_a{i},'-x',1:mv_power(i),res_power{i},'-s');
hold on
semilogy(1:mv_ae(i),res_ae{i},'-h',1:mv_m(i),res_m{i},'-p');
xlabel('The number of matrix-vector products');
ylabel('Residual norms');
legend('FOM','GMRES','W-FOM','W-GMRES','PW-FOM','PW-GMRES','W-Arnoldi','Power','EXT-Arnoldi','MPIO');
end


if write_or_not
disp('write into file')
fid = fopen('.\example3_web-Stanford_newPC_newTol2m8.txt','w');
 fprintf(fid,'FOM\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time(i));
end
fprintf(fid,'\n');

 fprintf(fid,'GMRES\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_g(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_g(i));
end
fprintf(fid,'\n');

 fprintf(fid,'WFOM\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_w(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_w(i));
end
fprintf(fid,'\n');

 fprintf(fid,'WGMRES\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_wg(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_wg(i));
end
fprintf(fid,'\n');

 fprintf(fid,'PWFOM\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_p(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_p(i));
end
fprintf(fid,'\n');

 fprintf(fid,'PWGMRES\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_pg(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_pg(i));
end
fprintf(fid,'\n');

 fprintf(fid,'W-Arnoldi\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_a(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_a(i));
end
fprintf(fid,'\n');

 fprintf(fid,'Power\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_power(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_power(i));
end
fprintf(fid,'\n');

 fprintf(fid,'Arnoldi_ext\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_ae(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_ae(i));
end
fprintf(fid,'\n');

 fprintf(fid,'MPIO\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_m(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_m(i));
end
fprintf(fid,'\n');

fprintf(fid,'FOM_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'GMRES_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_g_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_g_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'WFOM_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_w_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_w_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'WGMRES_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_wg_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_wg_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'PWFOM_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_p_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_p_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'PWGMRES_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_pg_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_pg_mgs2(i));
end
fprintf(fid,'\n');

 fprintf(fid,'W-Arnoldi_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_a_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_a_mgs2(i));
end
fprintf(fid,'\n');
 fprintf(fid,'Arnoldi_ext_mgs2\n');
 fprintf(fid,'mv\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',mv_ae_mgs2(i));
end
  fprintf(fid,'\n');
  fprintf(fid,'solving_time\n');
for i = 1:length(alpha)
      fprintf(fid,'%d\r',solving_time_ae_mgs2(i));
end
fprintf(fid,'\n');
 fclose(fid);
end
