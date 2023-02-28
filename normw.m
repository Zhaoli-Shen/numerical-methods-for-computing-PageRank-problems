%% compute the weighted norm of x with repect to the weighting vector wt
function y=normw(x,wt)
% weighted_FOM_global;
y=sqrt(x'*(x.*wt));
end