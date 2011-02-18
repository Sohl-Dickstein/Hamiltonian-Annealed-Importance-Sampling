function logL = HAIS_logL( HAIS_opts, X, varargin )
% logL = HAIS_logL( HAIS_options, X, [addl parameters] )
%
% Estimates the average log likelihood of samples X using Hamiltonian
% Annealed Importance Sampling.
% 
% See HAIS.m for information on HAIS_opts.
% See HAIS_examples.m for example usage.
%
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)
% Copyright 2011 Jascha Sohl-Dickstein, Jack Culpepper

    EN = getField( HAIS_opts, 'E', 0 );
    E = EN( X, varargin{:} );
    Eavg = mean(E);
    logZ = HAIS( HAIS_opts, varargin{:} );
    logL = -Eavg - logZ;
    