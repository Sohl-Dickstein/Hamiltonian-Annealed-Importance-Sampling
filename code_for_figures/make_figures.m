addpath(genpath('..'))

% the number of intermediate distributions to use
N_range = 10.^[1:0.05:6];
N_range = round(10.^[1:0.1:5.5]);

szd = 10; % number of dimensions
sze = szd; % number of experts -- same as number of dimensions for analytical log likelihood calculation
szb_train = 1000; % number of training datapoints
szb_hais = 10; % number of HAIS particles

opts = [];
opts.DataSize = szd; % the number of data dimensions
opts.BatchSize = szb_hais; % the number of HAIS particles
opts.epsilon = .5;
opts.E = @E_POE_studentt; % the energy function
opts.dEdX = @dEdX_POE_studentt; % the gradient of the energy function with
opts_array = {};
ii = 1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS - \gamma=1';
opts_array{ii}.ReducedFlip = 0;
opts_array{ii}.beta = 1;
ii = ii+1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS';
opts_array{ii}.ReducedFlip = 0;
ii = ii+1;
opts_array{ii} = opts;
opts_array{ii}.description = 'HAIS - Reduced Flip';
opts_array{ii}.ReducedFlip = 1;

% gether the descriptions into a more convenient datatype
descs = {};
for ii = 1:length(opts_array)
    descs{ii} = opts_array{ii}.description;
end


% generate training data
X = randn( szd, szb_train );
% initialize model parameters
theta_init = randn( sze, szd+1 ) / sqrt( szd+1 );
% train a model on the data
minf_opts = [];
%theta = minFunc( @L_dL_studentt, theta_init(:), minf_opts, X );
theta = [eye( sze ), log( ones(sze,1) )];
logL_true = -L_dL_studentt( theta, X );

logL = {};
logweights = {};
for ii = 1:length(opts_array)
    logL{ii} = NaN*zeros( 1, length(N_range) );
end

E = E_POE_studentt( X, theta );
for N_ind = randperm( length(N_range) )
    N = N_range(N_ind);
    for ii = 1:length(opts_array)
        opts_array{ii}.N = N;
        [logZ_l, logweights_l, X_l, P_l] = HAIS( opts_array{ii}, theta );
        logL_l = -E - logZ_l;
        logL{ii}(N_ind) = mean(logL_l);
    end
    
    sfigure(1); clf;
    sty = {'r.', 'g*', 'b+', 'co', 'y.', 'k.' };
    semilogx( [min(N_range), max(N_range)], [logL_true, logL_true], 'k--' );
    hold on;
    for ii = 1:length(opts_array)
        semilogx( N_range, logL{ii}, sty{ii} );
    end
    legend( {'True value', descs{:}} );
    title( 'Estimated log likelihood vs. number of intermediate distributions' );   
    drawnow;
end

return

if ~exist('do_not_reset') || do_not_reset == 0
    logZ_ais = zeros(1, length(N_range));
    logweights = zeros(opts.BatchSize, length(N_range) );
    loglike_ais = zeros(Btest, length(N_range));
    logZ_ais_gauss = zeros(1, length(N_range));
    logweights_gauss = zeros(opts.BatchSize, length(N_range) );
    loglike_ais_gauss = zeros(Btest, length(N_range));
    loglike_ais_gauss_mean = zeros(1, length(N_range));
    loglike_ais_mean = zeros(1, length(N_range));
    logZ_ais_b1 = zeros(1, length(N_range));
    logweights_b1 = zeros(opts.BatchSize, length(N_range) );
    loglike_ais_b1 = zeros(Btest, length(N_range));
end
do_not_reset = 0;


opts_b1 = opts;
opts_b1.beta = 1;


E = opts.E( Xtest, phi, more_args{:} );

if L == M
    switch model
        case 'studentt'
            trueL_best = -L_dL_studentt( phi_trueL(:), Xtest );
            trueL = -L_dL_studentt( phi(:), Xtest );
        case 'laplace'
            trueL_best = -L_dL_laplace( phi_trueL(:), Xtest );
            trueL = -L_dL_laplace( phi(:), Xtest );
    end
end


for i = 1:length(N_range)
    %% number of intermediate distributions for AIS
    opts.N = ceil(N_range(i));
    opts_b1.N = ceil(N_range(i));

    t_start = tic();
       
    if loglike_ais_gauss_mean(i) == 0
        [logZ_ais(1,i), logweights(:,i), X_HAIS] = HAIS( opts, phi, more_args{:} );
        loglike_ais(:,i) = -E - logZ_ais(1,i);
        loglike_ais_mean(i) = mean(loglike_ais(:,i));

        %[logZ_ais_b1(1,i), logweights(:,i)] = logZ( opts_b1, phi, more_args{:} );
        loglike_ais_b1(:,i) = -E - logZ_ais_b1(1,i);
        loglike_ais_b1_mean(i) = mean(loglike_ais_b1(:,i));    

        %[logZ_ais_gauss(1,i), logweights_gauss(:,i)] = AIS_gauss( opts, phi, more_args{:} );
        loglike_ais_gauss(:,i) = -E - logZ_ais_gauss(1,i);
        loglike_ais_gauss_mean(i) = mean(loglike_ais_gauss(:,i));
    end
    
    t_c = toc(t_start);
    
    fprintf('T %07d Log likelihood via AIS: %f in %f sec\n', opts.N, loglike_ais_mean(i), t_c);
    
    if 0
        sfigure(28);
        hist(logweights(:,1:i),20);
        %    legend( t_range(1:i ) );
        title( 'log weights' );
    end

    sfigure(20);
    semilogx(N_range(1:i), loglike_ais_mean(1:i), '.' );
    hold on;
    semilogx(N_range(1:i), loglike_ais_b1_mean(1:i), 'g.' );
    semilogx(N_range(1:i), loglike_ais_gauss_mean(1:i), 'r.' );
    hold off;
    grid on;
    legend('Estimated');
    title('Estimated log likelihood vs. number of intermediate distributions');
    xlabel('Number of intermediate distributions');
    ylabel('Log likelihood');
    if exist( 'trueL' )
        hold on;
        semilogx( [min(N_range), max(N_range)], [trueL, trueL], '--b' );
        semilogx( [min(N_range), max(N_range)], [trueL_best, trueL_best], '--g' );
        legend( 'MPF AIS', 'MPF AIS \beta=1', 'MPF AIS gauss', 'MPF true', 'best' );
        hold off;
    else
        legend( 'MPF AIS', 'MPF AIS \beta=1', 'MPF AIS gauss' );        
    end
    %    axis tight;

    if 0
        sfigure(16);
        semilogx(t_range(1:i), loglike_ais(:,1:i), '.-');
        title('Estimated log likelihood vs. number of intermediate distributions');
        xlabel('Number of intermediate distributions');
        ylabel('Log likelihood');
        axis tight;
    end

    drawnow;

    %if t_range(i) > 10^3.5
    %   eval(sprintf('save state/%s/matlab_up=%06d_ais.mat', paramstr, update)); 
    %end
end

opts_samp = opts;
opts_samp.sample=1;
[~,~, X_HAIS_samp] = HAIS( opts_samp, phi, more_args{:} );

%save POE_samp_laplace_bavm.mat V X_HAIS X_HAIS_samp