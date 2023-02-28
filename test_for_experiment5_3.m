%% written by Z.-L. Shen, SAU
%% related paper: preconditioned weighted FOM for PageRank computations (under review)

%% if the input all_res==1, the residual for every single iteration is recorded
%% else only the residual after every restart cycle is recorded
clear all
load wikipedia-20070206
G=Problem.A;
 alpha=[0.85,0.90,0.95,0.99];                                              % damping factors to test
m=8;                                                                       % restart steps
digits(16);
mg= max(sum(G,1));                                                                          % else only the residual after every restart cycle is recorded
epsilon=1e-16;
c=1.01*(1+3.03*epsilon);
maxit=1000;
gamma=0.1;                                                       % shift parameter to construct non-singular system for building preconditioner
l=1:10;                                                          % polynomial degree to test

%% construct the vectors to store the output informations
it=zeros(length(alpha),length(m));
it_w=it;
it_p=zeros(length(alpha),length(l));
it_p_uw=zeros(length(alpha),length(l));
solving_time=it;
solving_time_w=it;
solving_time_p=zeros(length(alpha),length(l));
solving_time_p_uw=zeros(length(alpha),length(l));
mv=it;
mv_w=it;
mv_p=zeros(length(alpha),length(l));
mv_p_uw=zeros(length(alpha),length(l));
res=cell(length(alpha),length(m));
res_w=cell(length(alpha),length(m));
res_p=cell(length(alpha),length(l));
res_p_uw=cell(length(alpha),length(l));
res1=it;
res1_w=it;
res1_p=zeros(length(alpha),length(l));
res1_p_uw=zeros(length(alpha),length(l));
res2=it;
res2_w=it;
res2_p=zeros(length(alpha),length(l));
res2_p_uw=zeros(length(alpha),length(l));
%% run the test
for i=1:length(alpha)
    tol=2*epsilon*(3.03+alpha(i)*c*mg)/(1-epsilon*(3.03+alpha(i)*c*mg));
for k=1:length(l)
[it_p(i,k),solving_time_p(i,k),mv_p(i,k),res_p{i,k},res1_p(i,k),res2_p(i,k)]=im_P_WFOM_pagerank_func(G,alpha(i),tol,m,maxit,l(k),gamma,0);
end
end

disp('write into file')
fid = fopen('.\example2_wikipedia-20070206_newtol.txt','w');
 fprintf(fid,'0.85\n');
 fprintf(fid,'it\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',it_p(1,i));
end
 fprintf(fid,'time\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',solving_time_p(1,i));
end

 fprintf(fid,'0.90\n');
 fprintf(fid,'it\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',it_p(2,i));
end
 fprintf(fid,'time\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',solving_time_p(2,i));
end

 fprintf(fid,'0.95\n');
 fprintf(fid,'it\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',it_p(3,i));
end
 fprintf(fid,'time\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',solving_time_p(3,i));
end

 fprintf(fid,'0.99\n');
 fprintf(fid,'it\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',it_p(4,i));
end
 fprintf(fid,'time\n');
for i = 1:length(l)
      fprintf(fid,'%d\r',solving_time_p(4,i));
end

 
 fclose(fid);
 
 plot(l,solving_time_p(1,:),'-*',l,solving_time_p(2,:),'-*',l,solving_time_p(3,:),'-*',l,solving_time_p(4,:),'-*');
 xlabel('The value of l');
ylabel('CPU time(s)');
legend('¦Á=0.85','¦Á=0.90','¦Á=0.95','¦Á=0.99');