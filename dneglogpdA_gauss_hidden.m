function dneglogpdA = neglogp_gauss_hidden( A, X, Jvisvis, Jhidhid, Jvishid  )
% return the gradient with respect to A for the negative log probability of each sample (column vector) in X and A.  A gaussian over both visible and hidden units.

    Xdiff = X - Jvishid * A;
    dneglogpdA = Jhidhid*A - Jvishid' * Jvisvis * Xdiff;
