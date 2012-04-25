% Here we demonstrate a variety of capabilities of the HAIS.m code,
% including:
%    - estimating the log partition function of a model p(x) = e^(-E(x))/Z
%    - estimating the expectation value of a function f(x) under a model
%      p(x)
%    - sampling from a distribution of interest p(x) via Hamiltonian Monte
%      Carlo
%    - estimating the log likelihood of data X under a model p(x)
%    - estimating the log likelihood of data X under a model p(x,a) with
%      hidden units a
%    - imposing linear constraints to x
%    - numerically verifying the user supplied gradient dEdX
%
% See included PDF HAIS.pdf for a description of the Hamiltonian Annealed
% Importance technique:
%    J Sohl-Dickstein, BJ Culpepper. Hamiltonian annealed importance sampling 
%    for partition function estimation. Redwood Technical Report. 2011.
%
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)
% Copyright 2011 Jascha Sohl-Dickstein, Jack Culpepper


%% Estimate the partition function for a Gaussian via HAIS.  The energy
% function and gradient function for the Gaussian are provided in E_gauss.m
% and dEdX_gauss.m.
fprintf( '\n\nEstimate the partition function for a Gaussian\n' );
d = 20; % the number of dimensions
% initialize parameter values randomly
% J is the coupling (inverse covariance) matrix for the Gaussian.
J = randn(d)/sqrt(d); J = J*J' + 0.1*eye(d);
% initialize options for HAIS
HAIS_opts = [];
HAIS_opts.DataSize = d; % the number of data dimensions
HAIS_opts.N = 1000; % the number of intermediate distributions to use (for most real applications, you will want 1E4 to 1E6 for this parameter)
HAIS_opts.E = @E_gauss; % the energy function
HAIS_opts.dEdX = @dEdX_gauss; % the gradient of the energy function with
                              % respect to X
% estimate the log partition function of the distribution
% described by HAIS_opts.E with parameters J via HAIS
logZ_estimate = HAIS( HAIS_opts, J );
% calculate the true partition function for the Gaussian
logZ_true = (d/2)*log(2*pi) - (1/2) * log(det(J));
fprintf( '%d dimensional Gaussian, true log partition function %f, estimated log partition function %f\n', d, logZ_true, logZ_estimate );



%% Here we use the HAIS code as a Hamiltonian Monte Carlo sampler, and
% sample from the distribution described by HAIS_opts.E and HAIS_opts.dEdX
% (in this case, the distribution is the same Gaussian as in the previous
% examples)
fprintf( '\n\nUse HAIS as a Hamiltonian Monte Carlo sampler, check covariance matrix of resulting samples\n' );
HAIS_opts.Sample = 1; % turn on sampling mode
nsamples = 1000; HAIS_opts.BatchSize = nsamples; % set the number of samples
[~,~,Xsamp] = HAIS( HAIS_opts, J ); % get samples
Csamp = Xsamp * Xsamp'/nsamples; % calculate the covariance matrix for samples
Cmodel = inv(J); % and the analytic covariance matrix for the model
% calculate the square error in the samples' covariance matrix
C_err = sqrt(mean((Csamp(:) - Cmodel(:)).^2));
fprintf( '%d samples from a %d dimensional Gaussian, %f rms error in covariance matrix\n', nsamples, d, C_err );
HAIS_opts.Sample = 0; % turn sampling mode back off


%% Here we calculate the covariance matrix - the expectation value of (x_i
% x_j) under p(x) = exp(-E)/Z - using the returned importance sampling
% weights rather than samples from p(x).
% Unlike for sampling, where it's very difficult to prove that mixing has
% occured, the importance weight estimate is guaranteed to always be
% unbiased ...  It may sometimes have larger error though, since unlike
% straight sampling, it spends most of its time annealing towards the right
% distribution, rather than sampling from the right distribution.
fprintf( '\n\nEstimate covariance matrix (expectation of x_i x_j under p(x) ) using importance weights\n' );
% get log importance weights (logW) and corresponding samples (X)
[~,logW,X] = HAIS( HAIS_opts, J );
% the importance weights can be very large ... subtract off their maximum value before exponentiating to avoid numerical problems
logW = logW - max(logW);
W = exp(logW); % the importance weights
% the covariance matrix is the weighted average of x_i x_j, using the importance weights
Cimportance = X * diag(W) * X' / sum(W);
C_err = sqrt(mean((Cimportance(:) - Cmodel(:)).^2));
fprintf( '%d importance samples for a %d dimensional Gaussian, %f rms error in estimated covariance matrix\n', nsamples, d, C_err );
HAIS_opts.BatchSize = 100; % set the number of particles back to its default value

% visually compare the covariance matrices found via sampling and importance
% weighting to the analytic (true) covariance matrix
fprintf( '\nDisplaying covariance matrices (see figure) ...' );
figure(1);
subplot( 1, 3, 1 ); imagesc( Csamp ); colorbar; axis image; title( 'Sample Covariance' );
subplot( 1, 3, 2 ); imagesc( Cmodel ); colorbar; axis image; title( 'Model Covariance' );
subplot( 1, 3, 3 ); imagesc( Cimportance ); colorbar; axis image; title( 'Importance Weighted Covariance' );
drawnow;


%% calculate the average log likelihood of test data under the Gaussian
% described by HAIS_opts.E and HAIS_opts.dEdX.
fprintf( '\n\nEstimate the average log likelihood of data under a Gaussian\n' );
X = randn( HAIS_opts.DataSize, 1000 ); % choose 1000 test data points randomly
logL_estimate = HAIS_logL( HAIS_opts, X, J );
% compare against true log likelihood
logL_true = -mean(E_gauss( X, J )) - logZ_true;
fprintf( '%d dimensional Gaussian, true average log likelihood %f nats, estimated average log likelihood %f nats\n', d, logL_true, logL_estimate );


%% Here we demonstrate placing constraints on the parameters.  We calculate
% the partition function for a non-negative Gaussian, described by the same
% energy function and gradient HAIS_opts.E and HAIS_opts.dEdX
fprintf( '\n\nCalculate the partition function for a non-negative Gaussian\n' );
% the inverse covariance matrix (or coupling matrix) for the Gaussian
J = diag(rand(d,1)+0.5);
% set the lower and upper bounds on each dimension - in this case 0 on the
% bottom and unbounded on top
HAIS_opts.Bounds = ones( HAIS_opts.DataSize, 1 ) * [0, Inf];
% estimate the partition function
logZ_estimate = HAIS( HAIS_opts, J );
% compare against true partition function.  This form for the partition
% function of a positive only Gaussian assumes that J is a diagonal matrix.
logZ_true = (d/2)*log(2*pi) - (1/2) * log(det(J)) - d*log(2);
fprintf( '%d dimensional non-negative Gaussian, true log partition function %f, estimated log partition function %f\n', d, logZ_true, logZ_estimate );


%% Here we estimate the average log likelihood of test data under a model
% with both visible and hidden (auxiliary) units.  We use a gaussian with
% both visible and hidden units (This can be thought of as a factor analysis
% model.  See neglogp_gauss_hidden.m for the exact model form.)  Estimating
% the log likelihood of a generative model involves estimating the log
% likelihood of each test data point separately
fprintf( '\n\nEstimate the log likelihood of a generative model (one with hidden units) - Gaussian prior over hidden units, gaussian conditional over X.  This involves separately calculating the log likelihood of all 10 test datapoints.\n' );
dvis = 10;
dhid = 20; % the number of *hidden* variables
% initialize parameter values randomly
Jvisvis = randn(dvis, dvis)/sqrt(dvis); Jvisvis = Jvisvis*Jvisvis' + 0.1*eye(dvis); % visible to visible coupling
Jhidhid = randn(dhid, dhid)/sqrt(dhid); Jhidhid = Jhidhid*Jhidhid' + 0.1*eye(dhid); % hidden  to hidden  coupling
Jvishid = randn(dvis, dhid)/sqrt(dvis+dhid); % visible to hidden  coupling
% initialize HAIS parameters
HAIS_opts = [];
HAIS_opts.DataSize = dhid; % the number of auxiliary data dimensions
HAIS_opts.N = 1000; % the number of intermediate distributions to use (for most real applications, you will want 1E4 to 1E6 for this parameter)
HAIS_opts.E = @neglogp_gauss_hidden;  % the negative log probability of the joint distribution over visible and hidden variables
HAIS_opts.dEdX = @dneglogpdA_gauss_hidden; % the gradient of the energy function with respect to auxiliary units A
% generate some random test data for the visible units
X = randn( dvis, 10 );
% estimate the log likelihood
[logL_estimate, logL_estimate_sample] = HAIS_logL_aux( HAIS_opts, X, Jvisvis, Jhidhid, Jvishid );
% compare against true log likelihood - this involves calculating the
% coupling matrix (inverse covariance matrix) for the gaussian distribution
% over the visible units X after marginalizing out the hidden units A
Q = inv(Jhidhid + Jvishid'*Jvisvis*Jvishid)*Jvishid'*Jvisvis;
Jmarginal = Jvisvis - Q'*Q;
% the log probability of each sample under the model
logL_true_sample = -(1/2)*sum( X.*(Jmarginal*X), 1) - (dvis/2)*log(2*pi) + (1/2)*log(det(Jmarginal));
% the average log likelihood of the model given all the samples
logL_true = mean(logL_true_sample);
fprintf( 'Gaussian with %d visible and %d hidden units, true average log likelihood %f nats, estimated average log likelihood %f nats\n', dvis, dhid, logL_true, logL_estimate );


%% Here we show how HAIS.m can be used to numerically verify the gradient
% function HAIS_opts.dEdX
fprintf( '\n\nNumerically test the gradient of the user supplied dEdX function\n' );
d = 3; % the number of dimensions
J = randn(d)/sqrt(d); % choose random parameter values
J = J*J'; % J is the inverse covariance matrix
% initialize HAIS parameters
HAIS_opts = [];
HAIS_opts.CheckGrad = 1;
HAIS_opts.DataSize = d; % the number of data dimensions
HAIS_opts.BatchSize = 5; % the number of particles to use
HAIS_opts.E = @E_gauss; % the energy function
HAIS_opts.dEdX = @dEdX_gauss; % the gradient of the energy function with respect to X - this is the function which is being tested
% call HAIS so as to compare numerical and analytical gradients
logZ_estimate = HAIS( HAIS_opts, J );
