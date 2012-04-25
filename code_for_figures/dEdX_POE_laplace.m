% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function dE = dEdX_POE( X, theta )
%function [dE, E] = dEdX_POE( X, theta )
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims);
	W = reshape( theta, [nexperts, ndims] );

        ff = W * X;
        sff = sign( ff );
        dE = W' * sff;
