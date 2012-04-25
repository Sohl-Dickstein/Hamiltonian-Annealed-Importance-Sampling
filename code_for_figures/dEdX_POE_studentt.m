% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function dE = dEdX_POE( X, theta )
%function [dE, E] = dEdX_POE( X, theta )
	% product of student-t experts
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims+1);
	W = reshape( theta, [nexperts, ndims+1] );
        alpha = exp(W(:,ndims+1));
        W = W(:, 1:ndims);

        ff = W * X;
        ff2 = ff.^2;
        off2 = 1 + ff2;
        %        lt = diag(alpha) * (ff./off2);        
        lt = bsxfun(@times, ff./off2, alpha);        
        dE = 2 * W' * lt;

        %        E = E_POE( theta, X ); %%% temporary for diagnostic purposes
        %figure(4);
        %plot(sort(E));
        