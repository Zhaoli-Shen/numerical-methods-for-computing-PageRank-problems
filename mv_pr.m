%% compute y=(I-A)x, with (I-A)=P+h*f for FOM methods
%% compute y=Ax, with A=P+h*f for Arnoldi methods
%% remark that A is the Google transition matrix
function y=mv_pr(x)
weighted_FOM_global;
y=P'*x+h*(f*x);

end