function [logZ, logweights, X, P] = HAIS( HAIS_opts, varargin )
% [logZ, logweights, X, P] = HAIS( HAIS_options, [addl parameters] )
%
% Performs Hamiltonian Annealed Importance Sampling, and returns an estimate
% of the log partition function (logZ), and the log of the importance
% weights (logweights), as well as the final state (X) and momentum (P) for
% all the particles.  The file HAIS_examples.m demonstrates several usage
% scenarios.  By default, a univariate Gaussian is used as the initial (easy
% to sample from and normalize) distribuion.
%
% HAIS_options is a structure containing all the settings for HAIS.  Sorted
% roughly from most to least useful, the supported fields are:
%   'DataSize': The number of data dimensions to use for AIS.
%   'E': The energy function for the distribution of interest.  The function
%       E should take arguments E( X, [addl parameters] ), and return a
%       vector containing the energy for each column of X.  X is a matrix of
%       size [DataSize, BatchSize].
%   'dEdX': The gradient of the energy function with respect to X.  The
%       function dEdX should take arguments dEdX( X, [addl parameters] ),
%       and return a matrix of the same size as X containing the derivative
%       of each column's energy.
%   'N': The number of intermediate distributions to use.
%       (default: 10000)
%   'BatchSize': The number of particles to use for AIS.  (default: 100)
%   'CheckGrad': Set to 1 to numerically check the gradient dEdX.
%       (default: 0)
%   'sample': Set to 1 to perform Hamiltonian Monte Carlo sampling of the
%       distribution described by E.  Samples are returned in X.  If this is
%       on, logZ and logweights will not be useful.  (The sampling method is
%       a single leapfrog step followed by partial momentum refreshment -
%       see Sections 5.2 and 5.3 of MCMC using Hamiltonian dynamics, R Neal,
%       Handbook of Markov Chain Monte Carlo, 2010). (default: 0)
%   'bounds': A matrix of size [DataSize, 2] containing lower and upper
%       bounds on X.  If 2 finite bounds are given, the initial distribution
%       will be uniform between the bounds.  If 1 finite bound is given, and
%       the other is infinite, the initial distribution will be a one sided
%       gaussian at the bound. (default: ones( DataSize, 1 ) * [-Inf, Inf])
%   'debug': Show debugging information and plots. 0: No display, 1: Print
%       single summary line, 2: Print an indicator every intermediate
%       distribution, 3: Display plots (default: 2)
%   'epsilon': The step size to use for the Hamiltonian dynamics steps
%       (default: 0.1)
%   'beta': The corruption rate for the momentum in the Hamiltonian
%       dynamics.  (default: set so as to replace half the momentum power
%       every 1/epsilon time steps)
%   'MixFrac': The mixing trajectory to use for interpolation between
%       the initial distribution and the model distribution.  A matrix of
%       size [N, 2], with the energy function at step n set to En =
%       MixFrac(n,1)*initE( X ) + MixFrac(n,2)*E(X, [addl parameters]
%       ). (default: [1-(0:N-1)'/(N-1), (0:N-1)'/(N-1)] )
%   'X0': The initial data state used for HAIS.  Should be a draw from the
%       n=0 distribution.  (default: randn( DataSize, BatchSize ) - but see
%       section on 'bounds' above)
%   'P0': The initial momentum used for HAIS.  (default: randn(
%       DataSize, BatchSize ))
%   'initE': Same format as 'E'.  Use this to set an alternative initial
%       distribution.  If this is changed, the HAIS_options inset below
%       should also be set.  (default: @E_HAIS_default)
%           'initdEdX': Same format as 'dEdX', but for the initial
%               distribution.  (default: @dEdX_HAIS_default)
%           'initlogZ': It should take arguments initlogZ( DataSize,
%               [addl parameters] ) and return the log partition function of
%               the initial (easy to sample from) distribution over X.
%               (default: @logZ_HAIS_default)
%           'X0': Described above - should be set to samples from initE(
%               X, [addl parameters] ).
%   'ParameterInterpolate': Rather than linearly interpolating the
%       energy function from a Gaussian to the distribution described by E(
%       X, [addl parameters] ), theta is linearly interpolated from an
%       initial value theta0 to a final value thetaN in the distribution
%       described by E( X, theta, [addl parameters]).  If this is turned on,
%       the HAIS_options inset below should also be set. (default: 0)
%           'theta0': The initial (easy to sample from) setting for
%               theta.  E( X, theta0, [addl parameters] ) should be easy to
%               sample from.
%           'thetaN': The final (distribution of interest) setting for
%               theta.  E( X, thetaN, [addl parameters] ) describes the
%               distribution of interest.
%           'X0': Described above - must be set to samples from E( X,
%               theta0, [addl parameters] ).
%           'initlogZ': Described above - must be set to return the
%               initial log partition function when called as given E( X,
%               theta0, [addl parameters] ).
%
% Example usage: See HAIS_examples.m
%
% See included PDF HAIS.pdf for a description of the Hamiltonian Annealed
% Importance technique:
%   J Sohl-Dickstein, BJ Culpepper. Hamiltonian annealed importance sampling 
%   for partition function estimation. Redwood Technical Report. 2011.
%
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)
% Copyright 2011 Jascha Sohl-Dickstein, Jack Culpepper
    
    t_forward = tic();

    %% load the parameters
    EN = getField( HAIS_opts, 'E', false );
    if islogical(EN)
        fprintf( '\n\nYou must specify an energy function, HAIS_opts.E.\n\n' );
        return;
    end
    dENdX = getField( HAIS_opts, 'dEdX', false );
    if islogical(dENdX)
        fprintf( '\n\nYou must specify an energy gradient function, HAIS_opts.dENdX.\n\n' );
        return;
    end
    logZ0 = getField( HAIS_opts, 'initlogZ', @logZ_HAIS_default );
    E0 = getField( HAIS_opts, 'initE', @E_HAIS_default );
    dE0dX = getField( HAIS_opts, 'initdEdX', @dEdX_HAIS_default );    
    param_interp = getField( HAIS_opts, 'ParameterInterpolate', 0 );
    interpType = getField( HAIS_opts, 'interpType', 'cos' );
    szb = getField( HAIS_opts, 'BatchSize', 100 );
    szd = getField( HAIS_opts, 'DataSize', -1 );
    if szd < 1
        if size( getField( HAIS_opts, 'X0', [] ), 1 ) < 1
            fprintf( '\n\nYou must specify the number of data dimensions, HAIS_opts.DataSize.\n\n' );
            return;
        end
    end
    bounds = getField( HAIS_opts, 'Bounds', ones( szd, 1 ) * [-Inf, Inf] );
    % collect indices where various combinations of upper and lower bounds apply
    upper_bounds_only = find( isfinite( bounds(:,2) ) & ~isfinite(bounds(:,1)) );
    lower_bounds_only = find( isfinite( bounds(:,1) ) & ~isfinite(bounds(:,2)) );
    both_bounds = find( isfinite( bounds(:,1) ) & isfinite(bounds(:,2)) );
    no_bounds = find( ~isfinite( bounds(:,1) ) & ~isfinite(bounds(:,2)) );
    if param_interp
        theta0 = getField( HAIS_opts, 'theta0', 0 );
        thetaN = getField( HAIS_opts, 'thetaN', 0 );
    end
    Debug = getField( HAIS_opts, 'Debug', 2 );
    N = getField( HAIS_opts, 'N', 10000 );
    epsilon = getField( HAIS_opts, 'epsilon', 0.1 );
    %% set the default beta value so as to replace half (or a fraction dut) of the momentum power per unit time
    dut = 0.5;
    beta = 1 - exp( log( dut ) * epsilon );
    beta = getField( HAIS_opts, 'beta', beta );
    % draw the initial state X0 from the initial distribution
    X0 = randn( szd, szb);
    % if there are bounds on the state sapce, adjust the initial distribution as appropriate
    X0( lower_bounds_only, : ) = bsxfun( @plus, abs(X0( lower_bounds_only, : )), bounds(lower_bounds_only, 1 ));
    X0( upper_bounds_only, : ) = bsxfun( @plus, -abs(X0( upper_bounds_only, : )), bounds(upper_bounds_only, 2 ));
    X0( both_bounds, : ) = bsxfun( @plus, bsxfun( @times, rand( length(both_bounds), szb ), bounds(both_bounds, 2 ) - bounds(both_bounds, 1 ) ), bounds(both_bounds, 1 ) );
    X0 = getField( HAIS_opts, 'X0', X0 ); % potentially overwrite the default inital state
    P0 = getField( HAIS_opts, 'P0', randn( szd, szb) ); % set the initial momentum
    szd = size( X0, 1 ); % number of data dimensions
    szb = size( X0, 2 ); % number of particles
    Sample = getField( HAIS_opts, 'Sample', 0 ); % act like Hamiltonian sampling code instead of AIS code
    % set the default annealing trajectory to linear interolation
    mix_frac_joint_arr = [1-(0:N-1)'/(N-1), (0:N-1)'/(N-1)];
    mix_frac_joint_arr = getField( HAIS_opts, 'MixFrac', mix_frac_joint_arr );
    if Sample % sample from the final distribution the entire time
        mix_frac_joint_arr = ones( N, 1 ) * [0, 1];
    end
    mix_frac0_arr = mix_frac_joint_arr(:,1);
    mix_frac1_arr = mix_frac_joint_arr(:,2);

    % put the bounds in a form to be more easily used for particle reflection
    lbounds_wide = bounds(:,1) * ones(1,szb);
    ubounds_wide = bounds(:,2) * ones(1,szb);    
    
    % initialize X and P.  X is state, P is momentum.
    X = X0;
    P = P0;
    
    CheckGrad = getField( HAIS_opts, 'CheckGrad', 0 );    
    if CheckGrad % numerically check the gradient
        cg_eps = 1e-8;
        cg_dEdX_analytic = dENdX(X, varargin{:});
        cg_dEdX_numerical = zeros(size(cg_dEdX_analytic));
        cg_E = EN(X, varargin{:});
        fprintf( '%12s %12s %12s\n', 'Analytic',  'Numerical',  'Difference' );
        for i = 1:prod(size(X))
            Xl = X;
            Xl(i) = Xl(i) + cg_eps;
            cg_El = EN(Xl, varargin{:});
            cg_diff = cg_El - cg_E;
            if sum( cg_diff ~= 0 ) > 1
                fprintf( 'Changing X for one particle changed the energy for a *different* particle!  Not good.\n' );
            end
            cg_dEdX_numerical(i) = sum(cg_diff)/cg_eps;
            fprintf( '%12.5e %12.5e %12.5e\n', cg_dEdX_analytic(i), cg_dEdX_numerical(i), cg_dEdX_analytic(i) - cg_dEdX_numerical(i) );
        end
        logZ=0;
        logweights=0;
        fprintf( '\nJust tested dEdX - will NOT run HAIS.  Set HAIS_opts.CheckGrad=0 to disable gradient check.\n' );
        return;
    end    

    % run the dynamics forward
    logw = 0;
    mix_frac0 = 1; % = mix_frac0_arr(1);
    mix_frac1 = 0; % = mix_frac1_arr(1);
    num_rej = 0;
    num_tot = 0;
    reweight = 0;
    if Debug > 2
        hist_dE = zeros(N,szb);
        hist_reweight = zeros(N,szb);
    end
    if param_interp
        theta = mix_frac0*theta0 + mix_frac1*thetaN;
        Em0 = EN(X, theta, varargin{:});
        E0n = 0;
        ENn = 0;
    else
        % the old and new energy function and gradients at this location
        E0n = E0(X, varargin{:}, bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds );
        ENn = EN(X, varargin{:});
        % the contribution to w and its gradient
        Em0 = mix_frac0*E0n + mix_frac1*ENn;
    end        
    for n = 2:N
        mix_frac0 = mix_frac0_arr(n);
        mix_frac1 = mix_frac1_arr(n);
        if param_interp
            theta = mix_frac0*theta0 + mix_frac1*thetaN;
            Em1 = EN(X, theta, varargin{:});
        else
            Em1 = mix_frac0*E0n + mix_frac1*ENn;
        end
        if Debug > 2
            hist_dE(n,:) =(-Em1 + Em0);
        end
        % accumulate the contribution to the importance weights from this step
        logw = logw - Em1 + Em0;
        
        % corrupt the momentum variable for langevin dynamics
        noise = randn( szd, szb );
        Pn  = -sqrt(1 - beta) * P + sqrt(beta) * noise;
        P = Pn;
        
        
        % run the langevin dynamics step
        P0 = P;
        X0 = X;
        E0n0 = E0n;
        ENn0 = ENn;
        % half step in the position
        X = X + epsilon/2 * P;
        % enforce the bounds for the first half step
        bd = find( X < lbounds_wide );
        X(bd) = lbounds_wide(bd) + (lbounds_wide(bd) - X(bd));
        P(bd) = -P(bd);
        bd = find( X > ubounds_wide );
        X(bd) = ubounds_wide(bd) + (ubounds_wide(bd) - X(bd));
        P(bd) = -P(bd);
        % full step in the momentum
        if param_interp
            dE = dENdX(X, theta, varargin{:});
        else
            dEm0dX = dE0dX(X, varargin{:}, bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds);
            dEmNdX = dENdX(X, varargin{:});
            dE = mix_frac0*dEm0dX + mix_frac1*dEmNdX;
        end
        P = P - epsilon * dE;
        % half step in the position
        X = X + epsilon/2 * P;
        % enforce the bounds for the second half step        
        bd = find( X < lbounds_wide );
        X(bd) = lbounds_wide(bd) + (lbounds_wide(bd) - X(bd));
        P(bd) = -P(bd);
        bd = find( X > ubounds_wide );
        X(bd) = ubounds_wide(bd) + (ubounds_wide(bd) - X(bd));
        P(bd) = -P(bd);

        % negate the momentum
        P = -P;

        E_orig = Em1;
        if param_interp
            %E_orig = EN(X0, theta, varargin{:});
            E_final = EN(X, theta, varargin{:});
        else
            %E_orig = mix_frac0*E0(X0, theta, varargin{:})  + mix_frac1*EN(X0, theta, varargin{:});
            E0n = E0(X, varargin{:}, bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds );
            ENn = EN(X, varargin{:});
            %        keyboard
            E_final = mix_frac0*E0n  + mix_frac1*ENn;
        end
        Em0 = E_final;

        % accept or reject the langevin step for each parameter
        delt_E = 0.5*sum(P.^2,1) + E_final - 0.5*sum(P0.^2,1) - E_orig;
        p_acc = exp( -delt_E );
        bd = p_acc < rand( 1, szb );
        P(:,bd) = P0(:,bd);
        X(:,bd) = X0(:,bd);
        Em0(bd) = E_orig(bd);
        E0n(bd) = E0n0(bd);
        ENn(bd) = ENn0(bd);
        num_rej = num_rej + sum(bd);
        num_tot = num_tot + size(P,2);        
            
        if Debug > 1
            fprintf('\rt %d / %d', n, N);
        end
    end
    fprintf('\n');

    % the estimate for logZ - adds the log partition function for the initial distribution to the log importance weights
    if param_interp
        logZ = logZ0( szd, theta0, varargin{:} ) + logw + reweight;
    else
        logZ = logZ0( szd, varargin{:}, bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds ) + logw + reweight;
    end
    
    logweights = logZ;
    % avoid numerical overflow - subtract a constant before exponentiating, and then add it again after logging
    C = max(logZ);
    logZ = log(sum( exp( logZ - C) )/szb) + C;
    
    t_forward = toc(t_forward);
    if Debug > 0
        fprintf( 'HAIS in %f sec, logZ %f, reject fraction %f', t_forward, logZ, num_rej / num_tot );
        if Sample
            fprintf( ' (Hamiltonian sampling only mode - set HAIS_opts.Sample=0 for importance sampling)' );
        end
        fprintf( '\n' );
    end
    if Debug > 2
            sfigure(12);
            clf;
            plot( hist_dE );
            title( 'weight contributions per sampling step' );
            xlabel( 'sampling step' );

        sfigure(13);
        plot( [mix_frac0_arr,mix_frac1_arr] );
        title( 'mix fraction per sampling step' );
        xlabel( 'sampling step' );
        legend( 'mix frac 0', 'mix frac 1' );
        drawnow;    
    end


    % these functions describe the default initial (n=1) distribution - a univariate gaussian if there are no bounds, a one sided gaussian with sigma=1 if there is one of a lower or upper bound, but not both, and a uniform distribution between the lower and upper bounds if both are finite.
    function E = E_HAIS_default( X, varargin )
        [bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds] = varargin{end-4:end};
        E = 0;        
        X(lower_bounds_only,:) = bsxfun( @plus, X(lower_bounds_only,:), - bounds(lower_bounds_only,1) );
        X(upper_bounds_only,:) = bsxfun( @plus, X(upper_bounds_only,:), - bounds(upper_bounds_only,2) );
        %X(both_bounds,:) = X(both_bounds,:) - bounds(both_bounds,2);
        E = E + 0.5 * sum( X(lower_bounds_only,:).^2, 1 );
        E = E + 0.5 * sum( X(upper_bounds_only,:).^2, 1 );
        E = E + 0.5 * sum( X(no_bounds,:).^2, 1 );
    end
    function dEdX = dEdX_HAIS_default( X, varargin )
        [bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds] = varargin{end-4:end};
        dE = zeros(size(X));
        X(lower_bounds_only,:) = bsxfun( @plus, X(lower_bounds_only,:), - bounds(lower_bounds_only,1) );
        X(upper_bounds_only,:) = bsxfun( @plus, X(upper_bounds_only,:), - bounds(upper_bounds_only,2) );
        %X(both_bounds,:) = X(both_bounds,:) - bounds(both_bounds,2);
        dEdX = X;
        dEdX(both_bounds,:) = 0; % if we're a uniform distribution between lower and upper bounds, then the energy is constant.
    end
    function logZ = logZ_HAIS_default( varargin )
        [bounds, upper_bounds_only, lower_bounds_only, both_bounds, no_bounds] = varargin{end-4:end};
        logZ = 0;
        logZ = logZ + sum( log( bounds(both_bounds,2) - bounds(both_bounds,1) ) ); % uniform energy 0 distribution between lower and upper bounds
        logZ = logZ + length( no_bounds ) * log( 2*pi )/2; % univariate gaussian
        logZ = logZ + length( lower_bounds_only ) * (log( 2*pi )/2 - log(2)); % one sided gaussian
        logZ = logZ + length( upper_bounds_only ) * (log( 2*pi )/2 - log(2)); % one sided gaussian
    end

end      

