function adnet_train_lair(x, numEpochs)
% ADNET_TRAIN Train the ADNet 
%
% Sangdoo Yun, 2017.

addpath('train/');
addpath(genpath('utils/'));
init_settings;
init_params_lair;
run(matconvnet_path);

rng(1004);

if nargin < 1
    x=30;
end
if nargin < 2
    numEpochs=30;
end
opts.numEpochs = numEpochs;

% Training stage 1: SL
opts.vgg_m_path = vgg_m_path;
tail=sprintf('LAIR%03d', x);
[net, all_vid_info] = adnet_train_SL_lair(opts,x,tail);
save(sprintf('./models/net_sl_%s.mat', tail), 'net')

% Training stage 2: RL
net = adnet_train_RL_lair(net, all_vid_info, opts, tail);
save(sprintf('./models/net_rl_%s.mat',tail), 'net')


