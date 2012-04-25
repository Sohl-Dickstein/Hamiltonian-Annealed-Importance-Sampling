% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [E, ff, off2, lff2] = E_POE_laplace( X, theta )
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims);
	W = reshape( theta, [nexperts, ndims] );
        
        ff = W * X;
        aff = abs(ff);        
	E = sum( aff, 1 );
 