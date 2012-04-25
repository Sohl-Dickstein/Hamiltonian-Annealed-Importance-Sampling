% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function dE = dEdThetaRescale_POE_studentt( X, theta, rscl, ff, off2, lff2 )
    
	% product of student-t experts
	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims+1);
	W = reshape( theta, [nexperts, ndims+1] );
        alpha = exp(W(:,ndims+1));
        W = W(:, 1:ndims);
        
        %        alpha(:) = mean(alpha); % *******************
        
        
        if ~exist( 'ff' )
            ff = W * X;
            ff2 = ff.^2;
            off2 = 1 + ff2;
            lff2 = log( off2 );
        end
        %alff2 = diag(alpha) * lff2;

        dEdalpha = (lff2 * rscl');
        
        
        %        dEdalpha(:) = mean(dEdalpha(:)); % ********************
        
        %        dEdalpha = lff2 * rscl(:);
        dEdlogalpha = alpha.* dEdalpha;

        %        lt = 2 * diag(alpha) * (ff./off2);        
        %        dEdW = lt * diag(rscl) * X';
        lt = bsxfun(@times, ff./off2, alpha);        
        %        dEdW = 2 * lt * diag(rscl) * X';
        dEdW = 2 * bsxfun( @times, lt, rscl ) * X';
        
        dE = zeros( nexperts, ndims+1 );
        dE( :, ndims+1 ) = dEdlogalpha;
        %        dE( :, ndims+1 ) = dEdalpha;
        dE( :, 1:ndims ) = dEdW;
        dE = dE(:);
        
