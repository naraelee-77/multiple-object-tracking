function net = init_model_mot_v3(dst_path, opts, fc4var, fc5var)
% INIT_MODEL Initialize ADNet using DagNN wrapper
% 
% Sangdoo Yun, 2017.

fprintf('Initializing model...\n');

src_model = opts.vgg_m_path;
to_save = true;

if exist(dst_path,'file')
    to_save = false;
    load(dst_path);    
    if isfield(net, 'net');
        net = net.net;
    end    
else
    % load conv layers
    load(src_model);
    % initialize networks
    simplenn_net = init_model_networks_mot_v2(layers, fc4var, fc5var);
    
    % modify networks to our model
    simplenn_net = modify_model_mot_v2(simplenn_net, opts);
    simplenn_net.layers = simplenn_net.layers(1:end-2);
    % convert to dagNN
    dagnn_net = dagnn.DagNN();
    dagnn_net = dagnn_net.fromSimpleNN(simplenn_net, 'CanonicalNames', true);
    net = dagnn_net;
    
    % add two fc6 layers (fc6_1, fc6_2)
    % FC6_1 layer: action output
    block1 = dagnn.Conv('size', [1 1 fc5var 3*opts.num_actions], 'hasBias', true);
    net.addLayer('fc6_1', block1, {'x16'}, {'all_predictions'}, ...
        {'fc6_1f', 'fc6_1b'});
    net.params(net.getParamIndex('fc6_1f')).value =  0.01 * randn(1,1,fc5var,3*opts.num_actions,'single');
    net.params(net.getParamIndex('fc6_1f')).learningRate = 10;
    net.params(net.getParamIndex('fc6_1b')).value =  zeros(1, 3*opts.num_actions, 'single');
    net.params(net.getParamIndex('fc6_1b')).learningRate = 20;
    
    % FC6_2 layer: binary classification output
    block2 = dagnn.Conv('size', [1 1 fc5var 3*2], 'hasBias', true);
    net.addLayer('fc6_2', block2, {'x16'}, {'all_prediction_scores'}, ...
        {'fc6_2f', 'fc6_2b'});
    net.params(net.getParamIndex('fc6_2f')).value =  0.01 * randn(1,1,fc5var,3*2,'single');
    net.params(net.getParamIndex('fc6_2f')).learningRate = 10;
    net.params(net.getParamIndex('fc6_2b')).value =  zeros(1, 3*2, 'single');
    net.params(net.getParamIndex('fc6_2b')).learningRate = 20;
    
    % slice threehot vectors into three onehot vectors
    slice = dagnn.Slice('slicePoint', [11 22]);
    net.addLayer('slice', slice, {'all_predictions'}, {'prediction1', 'prediction2', 'prediction3'});
    
    slice_score = dagnn.Slice('slicePoint', [2 4]);
    net.addLayer('slice_score', slice_score, {'all_prediction_scores'}, {'prediction_score1', 'prediction_score2', 'prediction_score3'});
    
    for id = 1:3
        % Loss 1 layer: softmax of action output (11 dim)
        net.addLayer(['loss' num2str(id)],dagnn.Loss('loss', 'softmaxlog'),{['prediction' num2str(id)], ['label' num2str(id)]},{['objective' num2str(id)]});

        % Loss 2 layer: softmax of object/background (2 dim)
        net.addLayer(['loss_score' num2str(id)],dagnn.Loss('loss', 'softmaxlog'),{['prediction_score' num2str(id)], ['label_score' num2str(id)]},{['objective_score' num2str(id)]});
       
        net.vars(net.getVarIndex(['objective' num2str(id)])).precious = 1;
        net.vars(net.getVarIndex(['objective_score' num2str(id)])).precious = 1;
    end
end

% % ------ Multi-Domain --------
% block = dagnn.Conv('size', [1 1 fc5var 3*opts.num_actions], 'hasBias', true);
% dummy_nets = {};
% for i = 1 : opts.num_videos
%     dmnet = dagnn.DagNN();
%     layer_name = ['v' num2str(i)];    
%     dmnet.addLayer(layer_name, block, {'x16'}, {'prediction'}, {'fc6_1f', 'fc6_1b'});
%     f_idx = net.getParamIndex('fc6_1f');
%     b_idx = net.getParamIndex('fc6_1b');
%     dmnet.params(dmnet.getParamIndex('fc6_1f')).value = 0.01 * randn(1,1,fc5var,3*opts.num_actions,'single');
%     dmnet.params(dmnet.getParamIndex('fc6_1f')).learningRate = 10;
%     dmnet.params(dmnet.getParamIndex('fc6_1b')).value = zeros(1, 3*opts.num_actions, 'single');
%     dmnet.params(dmnet.getParamIndex('fc6_1b')).learningRate = 20;
%     dummy_nets{i} = dmnet;
% end

% % scoring network
% block = dagnn.Conv('size', [1 1 fc5var 3*2], 'hasBias', true);
% dummy_score_nets = {};
% for i = 1 : opts.num_videos
%     dmnet = dagnn.DagNN();
%     layer_name = ['vs' num2str(i)];    
%     dmnet.addLayer(layer_name, block, {'x16'}, {'prediction_score'}, {'fc6_2f', 'fc6_2b'});
%     f_idx = net.getParamIndex('fc6_2f');
%     b_idx = net.getParamIndex('fc6_2b');
%     dmnet.params(dmnet.getParamIndex('fc6_2f')).value = 0.01 * randn(1,1,fc5var,3*2,'single');
%     dmnet.params(dmnet.getParamIndex('fc6_2f')).learningRate = 10;
%     dmnet.params(dmnet.getParamIndex('fc6_2b')).value = zeros(1, 3*2, 'single');
%     dmnet.params(dmnet.getParamIndex('fc6_2b')).learningRate = 20;
%     dummy_score_nets{i} = dmnet;
% end

if to_save
    save(dst_path, 'net');
end
