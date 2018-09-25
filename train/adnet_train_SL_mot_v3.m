function [net, all_vid_info] = adnet_train_SL_mot_v3(opts, tail, fc4var, fc5var)
% ADNET_TRAIN_SL Train the ADNet with supervised learning
% 
% Sangdoo Yun, 2017.

% ========================================================
% Set train DB
% ========================================================
% train_videos = get_train_videos(opts);
% opts.num_videos = numel(train_videos.video_names);

vid_path='../2DMOT2015/train';
vid_names=dir(vid_path);
vid_names=vid_names(~contains({vid_names.name}, '.'));
opts.num_videos = numel(vid_names);

% ========================================================
% Train - supervised learning
% ========================================================

% Constructing database: For each video and for each ID, make database of
% vid_info, which contains video name, image files, gt bboxes.
% Also make train_db, which also has pos and neg examples, action labels and score labels
fprintf('Construct database. \n');
% train_db_path = sprintf('./models/train_db_%s', tail);
train_db_path = './models/train_db_mot.mat';
[train_db, all_vid_info] = make_train_db(train_db_path, vid_path, vid_names, opts);
opts.num_videos=numel(train_db);

if exist(sprintf('./models/net_sl_%s', tail), 'file')
    fprintf('Supervised Learning completed.\n');
    load(sprintf('./models/net_sl_%s', tail), 'net')
    return;
end

% ========================================================
% Init model
% ========================================================

% initialize model with imagenet-vgg-m-conv1-3.mat, add fc layers
net = init_model_mot_v3(sprintf('./models/net_init_%s', tail), opts, fc4var, fc5var);
% return;
%
tic_train = tic;
train_cost = zeros(opts.numEpoch, numel(train_db));
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

opts.frameBatch = 5;
opts.minibatch_size = 128;

fprintf('Starting Supervised Learning...\n\n');
for epoch = 1 : opts.numEpoch
    for i = 1 : numel(train_db)
        % move to gpu and get dummy net layer, fc layer
        net.move('gpu');
        
%         dummy_net = dummy_nets{i};
%         dummy_net.move('gpu');
%         layer_name = ['v' num2str(i)];  
%         fc_layer = dummy_net.getLayer(layer_name);    
%         net.params(net.getParamIndex(fc_layer.params{1})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value;
%         net.params(net.getParamIndex(fc_layer.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value;
%         
%         % same as above for dummy score nets, fc layer score
%         dummy_score_net = dummy_score_nets{i};
%         dummy_score_net.move('gpu');
%         layer_name = ['vs' num2str(i)];  
%         fc_layer_score = dummy_score_net.getLayer(layer_name);    
%         net.params(net.getParamIndex(fc_layer_score.params{1})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value;
%         net.params(net.getParamIndex(fc_layer_score.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value;
        
        % shuffle frames during training
        frame_batch_shuffled = randperm(numel(train_db{i}));

        for j = 1 : min (numel(train_db{i}), opts.frameBatch)
            
            % get current image, data from train_db, check if grayscale
            train_img = imread(train_db{i}(frame_batch_shuffled(j)).img_path);
            bboxes = train_db{i}(frame_batch_shuffled(j)).bboxes;
            action_labels = train_db{i}(frame_batch_shuffled(j)).labels;
            score_labels = train_db{i}(frame_batch_shuffled(j)).score_labels;
            if(size(train_img,3)==1), train_img = cat(3,train_img,train_img,train_img); end
            
            % look at just the portion of the image from pos and neg examples
            imgs = get_masked_image_v3(train_img, bboxes);
            imgs = single(imgs);
            imgs = gpuArray(imgs);
            
            % from ground truth, get best action (translation/scale/etc)
            for id = 1:3
                inds = find(action_labels(:,:,id));
                [labels(:,id),~] = ind2sub(size(action_labels(:,:,id)), inds);
            end
            labels = gpuArray(labels);
            score_labels = gpuArray(score_labels);
            
            % training batches from one image
            % for action network
%             net.setExecutionOrder(exec_order_action);
            num_data = size(labels, 1);
            num_batches = num_data / opts.minibatch_size;
            batch_shuffled = randperm(num_data);
            for k = 1 : num_batches + 1
                % get random batch
                batch_start =  (k-1)*opts.minibatch_size+1 ;
                batch_end = min(num_data, k*opts.minibatch_size) ;
                curr_batch_idx = batch_shuffled(batch_start:batch_end);
                
                % evaluate performance for this single action chosen.
                % calculate loss and cost
                inputs = {'input', imgs(:,:,:,curr_batch_idx), ...
                    'label1', labels(curr_batch_idx,1), ...
                    'label2', labels(curr_batch_idx,2), ...
                    'label3', labels(curr_batch_idx,3), ...
                    'label_score1', score_labels(curr_batch_idx,1), ...
                    'label_score2', score_labels(curr_batch_idx,2), ...
                    'label_score3', score_labels(curr_batch_idx,3), ...
                    };
                outputs = {'objective1', 1, 'objective2', 1, 'objective3', 1, ...
                    'objective_score1', 1, 'objective_score2', 1, 'objective_score3', 1};        
                net.eval(inputs, outputs) ;
                [state] = accumulate_gradients_dagnn(state, net, opts.train, opts.minibatch_size);
                
                avg_loss = 0;
                for id = 1:3
                    curr_loss = gather(net.getVar(['objective' num2str(id)]).value); 
                    avg_loss = avg_loss + sum(curr_loss);
                end
                avg_loss = avg_loss / 3;
                
                train_cost(epoch,i) = train_cost(epoch,i) + avg_loss;
            end
             
        end
        % print info
        train_cost(epoch,i) = train_cost(epoch,i) / opts.minibatch_size;
        fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d, cost: %f \n', ... 
                toc(tic_train), epoch, opts.numEpoch, i, numel(train_db), train_cost(epoch,i));
            
%         % move back to cpu and save dummy nets and dummy score nets for next layer
%         dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value = net.params(net.getParamIndex(fc_layer.params{1})).value;
%         dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value = net.params(net.getParamIndex(fc_layer.params{2})).value;
%         dummy_net.move('cpu');
%         dummy_nets{i} = dummy_net;
%         
%         dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value = net.params(net.getParamIndex(fc_layer_score.params{1})).value;
%         dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value = net.params(net.getParamIndex(fc_layer_score.params{2})).value;
%         dummy_score_net.move('cpu');
%         dummy_score_nets{i} = dummy_score_net;
    end
end

% FINALIZE TRAINING
net.move('cpu');
% net.params(net.getParamIndex(fc_layer.params{1})).value = 0.01 * randn(1,1,2048,3*opts.num_actions,'single');
% net.params(net.getParamIndex(fc_layer.params{2})).value = zeros(1, 3*opts.num_actions, 'single');
% net.params(net.getParamIndex(fc_layer_score.params{1})).value = 0.01 * randn(1,1,2048,3*2,'single');
% net.params(net.getParamIndex(fc_layer_score.params{2})).value = zeros(1, 3*2, 'single');
% 
% save(sprintf('./models/net_dummy_%s',tail), 'dummy_nets', 'dummy_score_nets', '-v7.3');

