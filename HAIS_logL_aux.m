function [logL_avg, logL_all] = HAIS_logL_aux( HAIS_opts, X, varargin )
% [logL_avg, logL_all] = HAIS_logL_aux( HAIS_options, X, [addl parameters] )
%
% Estimates the average log likelihood (logL_avg) of samples X using
% Hamiltonian Annealed Importance Sampling under a model p( x, a ) with
% auxiliary variables a.  Also returns an array containing the estimated log
% likelihood of each individual sample, (logL_all).
% 
% There are several changes to the parameters, compared to models without
% auxiliary variables.  Several entries in HAIS_opts take on the following
% meanings:
%   'DataSize': This should be the number of *auxiliary* variables A.    
%   'E': This is now the *negative log probability* of the distribution of
%        interest, p( x, a ).  E will be called as E( A, X, [addl
%        parameters] ). X is a matrix of samples whose log likelihood will
%        be calculated.  X has size [#visible units, BatchSize], and A is a
%        matrix of size [DataSize, BatchSize].  To emphasize again, E should
%        be a function such that
%            E( A, X, [addl parameters] ) = -log[ p( X, A ) ]
%        and should include the log of the normalization constant.  E should
%        return a vector containing the energy for each column of X and A.
%   'dEdX': This is the gradient of the negative log probability - the
%        function in 'E' - with respect to the auxiliary variables *A*.  The
%        function dEdX should take arguments dEdX( A, X, [addl parameters]
%        ), and return a matrix of the same size as A containing the
%        derivative of each column's energy with respect to a.
%
% See HAIS.m for more information on HAIS_opts.
% See HAIS_examples.m for example usage.
%
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)
% Copyright 2011 Jascha Sohl-Dickstein, Jack Culpepper

    szb = getField( HAIS_opts, 'BatchSize', 100 );    
    Debug = getField( HAIS_opts, 'Debug', 2 );
    
    nsamples = size(X,2);
    logL_all = zeros(nsamples, 1);
    for i = 1:nsamples % calculate the log likelihood for each of the samples independently
        if Debug > 0
            fprintf( 'Calculating log likelihood of sample %d / %d\n', i, nsamples );
        end
        Xl = X(:,i);
        logL_all(i) = HAIS( HAIS_opts, Xl, varargin{:} );
    end
    logL_avg = mean( logL_all );
    