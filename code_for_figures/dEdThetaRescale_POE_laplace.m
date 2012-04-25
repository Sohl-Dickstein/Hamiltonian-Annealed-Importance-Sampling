% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function dE = dE_POE( X, theta, rscl, ff )    
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims);
	W = reshape( theta, [nexperts, ndims] );        
        if ~exist( 'ff' )
            ff = W * X;
        end
        sff = sign( ff );
        dEdW = bsxfun( @times, sff, rscl ) * X';
        dE = dEdW(:);
        
