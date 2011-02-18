function dEdX = dEdX_gauss( X, J )
% return the gradient with respect to X for the energy of each sample
% (column vector) in X for a Gaussian with inverse covariance matrix
% (coupling matrix) J

    dEdX = 0.5*J*X + 0.5*J'*X;
    