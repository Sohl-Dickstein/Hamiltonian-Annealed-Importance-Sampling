% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [E, ff] = E_POE( X, theta, O, lambda_edge )
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims+O);
	W = reshape( theta, [nexperts, ndims+O] );
        gamma = W(:,ndims+1:end);
        W = W(:,1:ndims);
        
        ff = W * X;
        %tff = tanh(ff);        
	%E = E_bins( gamma, tff );
        E = E_bins( gamma, ff );
 
        %% continue beyond the end with a gaussian
        bd = abs(ff) > 1-eps;
        E(bd) = E(bd) + lambda_edge*ff(bd).^2 - lambda_edge;
        
        E = sum( E );
 