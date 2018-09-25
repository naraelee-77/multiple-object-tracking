function adnet_train_mot_v3(fc4var, fc5var)
% ADNET_TRAIN Train the ADNet 
%
% Sangdoo Yun, 2017.

addpath('train/');
addpath(genpath('utils/'));
init_settings;
init_params_mot_v2;
run(matconvnet_path);
rng(1004);

tail = sprintf('mot_v3_%d_%d.mat', fc4var, fc5var);

% Training stage 1: SL
opts.vgg_m_path = vgg_m_path;
[net, all_vid_info] = adnet_train_SL_mot_v3(opts, tail, fc4var, fc5var);
save(sprintf('./models/net_sl_%s', tail), 'net')

% Training stage 2: RL
net = adnet_train_RL_mot_v3(net, all_vid_info, opts, tail, fc5var);
save(sprintf('./models/net_rl_%s', tail), 'net')


