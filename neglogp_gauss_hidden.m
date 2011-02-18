function neglogp = neglogp_gauss_hidden( A, X, Jvisvis, Jhidhid, Jvishid  )
% return the negative log probability for each sample (column vector) in X and A.  A gaussian over both visible and hidden units.

    nhid = size(A,1);
    nvis = size(X,1);
    
    Ea = (1/2)*sum(A.*(Jhidhid*A));
    Za = (sqrt(2*pi)^nhid * det(Jhidhid)^(-1/2));
    logpa = -Ea - log(Za); % the gaussian prior over a, p(a)

    Xdiff = X - Jvishid * A;
    Ex_a = (1/2)*sum(Xdiff.*(Jvisvis*Xdiff));
    Zx_a = (sqrt(2*pi)^nvis * det(Jvisvis)^(-1/2));
    logpx_a = -Ex_a - log(Zx_a); % the gaussian conditional over x, p(x|a)

    neglogp = -logpa - logpx_a;
