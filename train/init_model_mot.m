function [net, dummy_nets, dummy_score_nets] = init_model_mot(dst_path, opts)
% INIT_MODEL Initialize ADNet using DagNN wrapper
% 
% Sangdoo Yun, 2017.

src_model = opts.vgg_m_path;

if exist(dst_path,'file')
    load(dst_path);    
    if isfield(net, 'net');
        net = net.net;
    end    
else
    % load conv layers
    load(src_model);
    % initialize networks
    simplenn_net = init_model_networks(layers);
    % modify networks to our model
    simplenn_net = modify_model(simplenn_net, opts);
    simplenn_net.layers = simplenn_net.layers(1:end-2);
    
    % convert to dagNN
    dagnn_net = dagnn.DagNN();
    dagnn_net = dagnn_net.fromSimpleNN(simplenn_net, 'CanonicalNames', true);
    
    temp_id1 = dagnn_net.saveobj();
    temp_id2 = dagnn_net.saveobj();
    temp_id3 = dagnn_net.saveobj();
    
    temp_id1 = modify_dagnn(temp_id1, 1);
    temp_id2 = modify_dagnn(temp_id2, 2);
    temp_id3 = modify_dagnn(temp_id3, 3);
    
    temp_net.layers = [temp_id1.layers temp_id2.layers temp_id3.layers];
    temp_net.vars   = [temp_id1.vars   temp_id2.vars   temp_id3.vars  ];
    temp_net.params = [temp_id1.params temp_id2.params temp_id3.params];
    temp_net.meta = dagnn_net.meta;
    
    net = dagnn.DagNN.loadobj(temp_net);
    
    net.addLayer('concat_patch', dagnn.Concat('dim', 1), {'x16_id1', 'x16_id2', 'x16_id3'}, 'x16_total');
        
    for id = 1:3
        % add two fc6 layers (fc6_1, fc6_2)
        % FC6_1 layer: action output
        block1 = dagnn.Conv('size', [3 1 512 opts.num_actions], 'hasBias', true);
        net.addLayer(sprintf('fc6_1_id%d', id), block1, {'x16_total'}, {sprintf('prediction_id%d', id)}, ...
            {sprintf('fc6_1f_id%d', id), sprintf('fc6_1b_id%d', id)});
        net.params(net.getParamIndex(sprintf('fc6_1f_id%d', id))).value =  0.01 * randn(3,1,512,opts.num_actions,'single');
        net.params(net.getParamIndex(sprintf('fc6_1f_id%d', id))).learningRate = 10;
        net.params(net.getParamIndex(sprintf('fc6_1b_id%d', id))).value =  zeros(1, opts.num_actions, 'single');
        net.params(net.getParamIndex(sprintf('fc6_1b_id%d', id))).learningRate = 20;

        % FC6_2 layer: binary classification output
        block2 = dagnn.Conv('size', [3 1 512 2], 'hasBias', true);
        net.addLayer(sprintf('fc6_2_id%d', id), block2, {'x16_total'}, {sprintf('prediction_score_id%d', id)}, ...
            {sprintf('fc6_2f_id%d', id), sprintf('fc6_2b_id%d', id)});
        net.params(net.getParamIndex(sprintf('fc6_2f_id%d', id))).value =  0.01 * randn(3,1,512,2,'single');
        net.params(net.getParamIndex(sprintf('fc6_2f_id%d', id))).learningRate = 10;
        net.params(net.getParamIndex(sprintf('fc6_2b_id%d', id))).value =  zeros(1, 2, 'single');
        net.params(net.getParamIndex(sprintf('fc6_2b_id%d', id))).learningRate = 20;

        % Loss 1 layer: softmax of action output (11 dim)
        softmaxlossBlock1 = dagnn.Loss('loss', 'softmaxlog');
        net.addLayer(sprintf('loss_id%d', id),softmaxlossBlock1,{sprintf('prediction_id%d', id), sprintf('label_id%d', id)},{sprintf('objective_id%d', id)});

        % Loss 2 layer: softmax of object/background (2 dim)
        softmaxlossBlock2 = dagnn.Loss('loss', 'softmaxlog');
        net.addLayer(sprintf('loss_score_id%d', id),softmaxlossBlock2,{sprintf('prediction_score_id%d', id), sprintf('label_score_id%d', id)},{sprintf('objective_score_id%d', id)});
        
        net.vars(net.getVarIndex(sprintf('objective_id%d', id))).precious = 1;
        net.vars(net.getVarIndex(sprintf('objective_score_id%d', id))).precious = 1;
        
    end
    
end

% ------ Multi-Domain --------
block = dagnn.Conv('size', [3 1 512 opts.num_actions], 'hasBias', true);
dummy_nets = {};
for i = 1 : opts.num_videos
    dmnet = dagnn.DagNN();
    
    for id = 1:3
        layer_name = sprintf('v%d_id%d', i, id); 
    
        dmnet.addLayer(layer_name, block, {'x16_total'}, ...
            {sprintf('prediction_id%d', id)}, {sprintf('fc6_1f_id%d', id), sprintf('fc6_1b_id%d', id)});
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_1f_id%d', id))).value = 0.01 * randn(3,1,512,opts.num_actions,'single');
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_1f_id%d', id))).learningRate = 10;
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_1b_id%d', id))).value = zeros(1, opts.num_actions, 'single');
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_1b_id%d', id))).learningRate = 20;
    end
    
    dummy_nets{i} = dmnet;
end

% scoring network
block = dagnn.Conv('size', [3 1 512 2], 'hasBias', true);
dummy_score_nets = {};
for i = 1 : opts.num_videos
    dmnet = dagnn.DagNN();
    
    for id = 1:3
        layer_name = sprintf('vs%d_id%d', i, id);    
        dmnet.addLayer(layer_name, block, {'x16_total'}, ...
            {sprintf('prediction_score_id%d', id)}, {sprintf('fc6_2f_id%d', id), sprintf('fc6_2b_id%d', id)});
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_2f_id%d', id))).value = 0.01 * randn(3,1,512,2,'single');
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_2f_id%d', id))).learningRate = 10;
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_2b_id%d', id))).value = zeros(1, 2, 'single');
        dmnet.params(dmnet.getParamIndex(sprintf('fc6_2b_id%d', id))).learningRate = 20;
    end
    
    dummy_score_nets{i} = dmnet;
end

save(dst_path, 'net');
