% calculate negative log likelihood and gradient for product of experts with student's-t experts

function [L, dL] = L_dL_studentt( theta, X )

    	ndims = size(X, 1);
	nbatch = size(X, 2);
        nexperts = prod(size(theta)) / (ndims+1);
	W = reshape( theta, [nexperts, ndims+1] );
        theta = W;
        alpha = exp(W(:,ndims+1));
        W = W(:, 1:ndims);
    
        E = sum(E_POE_studentt( X, theta ));
        rscl = ones(1, nbatch);
        dE = dEdThetaRescale_POE_studentt( X, theta, rscl );
        E = E / nbatch;
        dE = dE/nbatch;
        
        nu = alpha*2-1;
        %        logZ = - log( gamma( (nu + 1) / 2 ) ) + 0.5*log( nu * pi ) + log( gamma( nu/2 ) ) - 0.5 * log(nu);
        logZ = - log( gamma( (nu + 1) / 2 ) ) + 0.5*log( pi ) + log( gamma( nu/2 ) );
        logZ = sum(logZ) - log(abs(det(W))); % contribution from the feedforward weight matrix
        
        dlogZdW = -inv(W)'; % log determinant contribution
        %if det(W) > 0
            %    dlogZdW = -inv(W)'; % log determinant contribution
            %else
            %dlogZdW = -inv(W)'; % log determinant contribution
            %end
                                         %nu
        dlogZdnu = -psi( (nu + 1) / 2 )/2 + psi( nu/2 )/2;
        dlogZdalpha = 2 * dlogZdnu;
        dlogZdlogalpha = alpha .* dlogZdalpha;
        
        dlogZ = zeros( nexperts, ndims+1 );
        dlogZ( :, ndims+1 ) = dlogZdlogalpha;
        dlogZ( :, 1:ndims ) = dlogZdW;
        dlogZ = dlogZ(:);
 
        dL = -dE - dlogZ;
        L = -E - logZ;

        L = -L; % *negative* log likelihood!
        dL = -dL;