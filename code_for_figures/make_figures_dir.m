addpath(genpath('..'))

% the number of intermediate distributions to use
N_range = 10.^[1:0.05:6];
N_range = round(10.^[1:0.01:4.5]);
%N_range = round(10.^[1:0.01:3.5]);
N_range = round(10.^[0:0.01:4]);

fig_num = ceil( rand() * 1e6 );

szd = 100; % number of dimensions
szd = 20; % number of dimensions
sze = szd; % number of experts -- same as number of dimensions for analytical log likelihood calculation
sze = szd*2;
szb_train = 10; % number of training datapoints
szb_hais = 1000; % number of HAIS particles

model = 'POE_studentt';
model = 'Dir_gauss';

opts = [];
opts.DataSize = sze; % the number of data dimensions
opts.BatchSize = szb_hais; % the number of HAIS particles
opts.epsilon = .1;
opts.E = @neglogp_gauss_hidden;  % the negative log probability of the joint distribution over visible and hidden variables
opts.dEdX = @dneglogpdA_gauss_hidden; % the gradient of the energy function with respect to auxiliary units A


opts_array = {};
ii = 0;
ii = ii+1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS - \gamma=1';
opts_array{ii}.ReducedFlip = 0;
opts_array{ii}.QuasiStatic = 0;
opts_array{ii}.beta = 1;
ii = ii+1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS';
opts_array{ii}.ReducedFlip = 0;
opts_array{ii}.QuasiStatic = 0;
%ii = ii+1;
%opts_array{ii} = opts;
%opts_array{ii}.description = 'HAIS SM';
%opts_array{ii}.ReducedFlip = 0;
%opts_array{ii}.QuasiStatic = 0;
%opts_array{ii}.SampleMeta = 1;
% ii = ii+1;
% opts_array{ii} = opts;
% opts_array{ii}.description = 'HAIS - QS abs';
% opts_array{ii}.ReducedFlip = 0;
% opts_array{ii}.QuasiStatic = 1;
ii = ii+1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS QS';
opts_array{ii}.ReducedFlip = 0;
opts_array{ii}.QuasiStatic = 2;
% ii = ii+1;
% opts_array{ii} = opts;
% opts_array{ii}.description = 'HAIS MR';
% opts_array{ii}.ReducedFlip = 0;
% opts_array{ii}.QuasiStatic = 0;
% opts_array{ii}.MomentumRedraw = 1;
% ii = ii+1;
% opts_array{ii} = opts;
% opts_array{ii}.description = 'HAIS QS MR';
% opts_array{ii}.ReducedFlip = 0;
% opts_array{ii}.QuasiStatic = 2;
% opts_array{ii}.MomentumRedraw = 1;
% ii = ii+1;
% opts_array{ii} = opts;
% opts_array{ii}.description = 'HAIS QS MR RF';
% opts_array{ii}.ReducedFlip = 1;
% opts_array{ii}.QuasiStatic = 2;
% opts_array{ii}.MomentumRedraw = 1;

% gether the descriptions into a more convenient datatype
descs = {};
for ii = 1:length(opts_array)
    descs{ii} = opts_array{ii}.description;
end


% generate training data
X = randn( szd, szb_train );


dvis = szd;
dhid = sze; % the number of *hidden* variables
% initialize parameter values randomly
Jvisvis = randn(dvis, dvis)/sqrt(dvis); Jvisvis = Jvisvis*Jvisvis' + 0.1*eye(dvis); % visible to visible coupling
Jhidhid = randn(dhid, dhid)/sqrt(dhid); Jhidhid = Jhidhid*Jhidhid' + 0.1*eye(dhid); % hidden  to hidden  coupling
hs = log(1e-3);
he = log(1e1);
Jhidhid = diag( exp(hs:((he-hs))/(sze-1):he ) );
Jvishid = randn(dvis, dhid)/sqrt(dvis+dhid); % visible to hidden  coupling
%Jvishid = zeros(dvis, dhid);

%Q = inv(Jhidhid + Jvishid'*Jvisvis*Jvishid)*Jvishid'*Jvisvis;
%Jmarginal = Jvisvis - Q'*Q;
Vvishid = Jvisvis*Jvishid;
Jmarginal = Jvisvis - Vvishid*inv(Jhidhid+Jvishid'*Jvisvis*Jvishid)*Vvishid';
% the log probability of each sample under the model
logL_true_sample = -(1/2)*sum( X.*(Jmarginal*X), 1) - (dvis/2)*log(2*pi) + (1/2)*log(det(Jmarginal));
% the average log likelihood of the model given all the samples
logL_true = mean(logL_true_sample);


logL = {};
t_est = {};
logweights = {};
for ii = 1:length(opts_array)
    logL{ii} = NaN*zeros( 1, length(N_range) );
    t_est{ii} = zeros( 1, length(N_range) );
end

max_t = 0;
min_t = 9999999999999;

for N_ind = randperm( length(N_range) )
    N = N_range(N_ind);
    for ii = 1:length(opts_array)
        opts_array{ii}.N = N;
        opts_array{ii}.EnergyStep = 10/N;
        if opts_array{ii}.QuasiStatic
            opts_array{ii}.N = 100*N;
        end
        t_start = tic();
        %[logZ_l, logweights_l, X_l, P_l] = HAIS( opts_array{ii}, theta );
        [logL_estimate, logL_estimate_sample] = HAIS_logL_aux( opts_array{ii}, X, Jvisvis, Jhidhid, Jvishid );

        t_est{ii}(N_ind) = toc( t_start );
        logL{ii}(N_ind) = logL_estimate;
    end
    for ii = 1:length(opts_array)
        max_t = max( [max_t, t_est{ii}(N_ind)] );
        min_t = min( [min_t, t_est{ii}(N_ind)] );
    end

    
    sfigure(fig_num); clf;
    sty = {'r.', 'g*', 'b+', 'co', 'y.', 'k.' };
    semilogx( [min_t, max_t], [logL_true, logL_true], 'k--' );
    hold on;
    for ii = 1:length(opts_array)
        semilogx( t_est{ii}, logL{ii}, sty{ii} );
    end
    legend( {'True value', descs{:}}, 'Location', 'Best' );
    title( 'Estimated log likelihood vs. number of intermediate distributions' );  
    ylabel( 'Log likelihood (nats)' );
    xlabel( 'Computation time (s)' );
    grid on;
    drawnow;
end

