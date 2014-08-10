% simple network to learn algebraic relations between 2 variables encoded
% using a self-organizing-map of sigma-pi units

%% SETUP ENV
clear all; close all; clc


%% INPUT DATA
% simple relation z = x+y
MIN_VAL_IN = 0.0;
MAX_VAL_IN = 1.0;
MIN_VAL_OUT = 0.0;
MAX_VAL_OUT = 2.0;
sensory_data.x = MIN_VAL_IN:0.1:MAX_VAL_IN;
sensory_data.y = MIN_VAL_IN:0.1:MAX_VAL_IN;
sensory_data.z = MIN_VAL_OUT:0.1:MAX_VAL_OUT;

% number of neurons in each population
N_NEURONS = 20;
% maximum initialization range of weights and activities in the populations
MIN_INIT_RANGE = -0.1;
MAX_INIT_RANGE = 0.1;
% learning rate
ETA = 0.01;
% number of training epochs
MAX_EPOCHS = 300000;
% neighborhood function width
SIGMA0  = N_NEURONS/2;
t = 1:MAX_EPOCHS;
% decrease the neighborhood function
SIGMAT = SIGMA0*exp(-t./(MAX_EPOCHS/log(SIGMA0)));
% normalalization factor for activities
NORM_ACT = 1;

% network structure (populations)
population_x = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));

population_y = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));

population_x_bar = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));

population_y_bar = struct('lsize', N_NEURONS, ...
    'A', rand(N_NEURONS, 1));

population_z = struct('lsize', N_NEURONS,...
    'A', rand(N_NEURONS, 1), ...
    'W', MIN_INIT_RANGE + (MAX_INIT_RANGE - MIN_INIT_RANGE)*rand(N_NEURONS, N_NEURONS, N_NEURONS));

% handler for changes in activity in output layer
delta_act = zeros(N_NEURONS, N_NEURONS);

%% NETWORK SIMULATION
for t = 1:MAX_EPOCHS
    % choose a random value from z sensory data
    val_z = MIN_VAL_OUT + (MAX_VAL_OUT - MIN_VAL_OUT)*rand;
    
    % sample randomly and input pair (x, y) such that x+y=z
    val_x = MIN_VAL_IN + (MAX_VAL_IN - MIN_VAL_IN)*rand;
    val_y = val_z - val_x;
    
    % produce the population code for (x, y) pair, post-synaptic
    population_x.A = population_encoder(val_x, MAX_VAL_IN, N_NEURONS);
    population_y.A = population_encoder(val_y, MAX_VAL_IN, N_NEURONS);
    
    % forward propagation of activities in populations x and y to z
    for idx = 1:N_NEURONS % in z population
        for jdx = 1:N_NEURONS % in x population
            for kdx = 1:N_NEURONS % in y population
               delta_act(idx, jdx) = population_z.W(idx, jdx, kdx)*population_x.A(jdx)*population_y.A(kdx);
            end
        end
        population_z.A(idx) = sum(delta_act(:));
    end
    
    % find the best matching unit
    [bmu_val, bmu_pos] = max(conv(population_z.A, gauss_kernel(NORM_ACT, N_NEURONS, SIGMAT(t))));
    
    % compute activations in the output population z
    population_z.A = gauss_kernel(NORM_ACT, bmu_pos, SIGMAT(t));
    
    % sample randomly a pair(x_bar, y_bar) such that x_bar+y_bar=z
    val_x_bar = MIN_VAL_IN + (MAX_VAL_IN - MIN_VAL_IN)*rand;
    val_y_bar = val_z - val_x_bar;
    
    % produce the population code for (x_bar, y_bar) pair, pre-synaptic
    population_x_bar.A = population_encoder(val_x_bar, MAX_VAL_IN, N_NEURONS);
    population_y_bar.A = population_encoder(val_y_bar, MAX_VAL_IN, N_NEURONS);
    
    % apply Hebbian learning to update weights
    for idx = 1:N_NEURONS % in z population
        for jdx = 1:N_NEURONS % in x population
            for kdx = 1:N_NEURONS % in y population
                population_z.W(idx, jdx, kdx) = population_z.W(idx, jdx, kdx) + ...
                    ETA*population_z.A(idx)*(population_x_bar.A(jdx)*population_y_bar.A(kdx) - population_z.W(idx, jdx, kdx));
            end
        end
    end
    
end % end of input dataset

%% VISUALIZATION
figure; set(gcf, 'color', 'w');
for idx = 1:N_NEURONS
    subplot(1, N_NEURONS, idx);
    Wm(:,:) = population_z.W(idx, :, :);
    pcolor(Wm); colorbar;
end