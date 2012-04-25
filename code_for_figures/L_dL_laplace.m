function [L, dL] = L_dL_studentt( theta, X )

    ndims = size(X, 1);
    nbatch = size(X, 2);
    nexperts = prod(size(theta)) / (ndims);
    W = reshape( theta, [nexperts, ndims] );
    
    E = sum(E_POE_laplace( X, theta ));
    rscl = ones(1, nbatch);
    dE = dEdThetaRescale_POE_laplace( X, theta, rscl );
    E = E / nbatch;
    dE = dE/nbatch;
    
    logZ = log(2) * ndims;
    logZ = logZ - log(abs(det(W))); % contribution from the feedforward weight matrix
        
    dlogZdW = -inv(W)'; % log determinant contribution
                                         %nu
    dlogZ = dlogZdW;
    dlogZ = dlogZ(:);
 
    dL = -dE - dlogZ;
    L = -E - logZ;
    
    L = -L; % *negative* log likelihood!
    dL = -dL;