function E = E_gauss( X, J )
% return the energy for each sample (column vector) in X for a Gaussian with
% inverse covariance matrix (coupling matrix) J
        
    E = 0.5*sum( X.*(J*X), 1 );