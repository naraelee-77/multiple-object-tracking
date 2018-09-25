function adnet_train_mot
% ADNET_TRAIN Train the ADNet 
%
% Sangdoo Yun, 2017.

addpath('train/');
addpath(genpath('utils/'));
init_settings;
init_params_mot;
run(matconvnet_path);

rng(1004);

% Training stage 1: SL
opts.vgg_m_path = vgg_m_path;
[net, all_vid_info] = adnet_train_SL_mot(opts);
% save('./models/net_sl_mot.mat', 'net')

% Training stage 2: RL
net = adnet_train_RL_mot(net, all_vid_info, opts);
save('./models/net_rl_mot.mat', 'net')


