function net = adnet_train_RL_mot_v3(net, all_vid_info, opts, tail, fc5var)
% ADNET_TRAIN_RL Train the ADNet with reinforcement learning
% 
% Sangdoo Yun, 2017.

% model from SL
% load(sprintf('./models/net_dummy_%s', tail)); % alg 1 line 1

% change layers around
net.addLayer('concat_',dagnn.Concat, {'x16', 'action_history'}, 'x16_ah');
net.addLayer('concat__',dagnn.Concat, {'x16', 'action_history'}, 'x16_ah');

index = net.getLayerIndex('fc6_1');
net.layers(index+1:end-1) = net.layers(index:end-2); % make room for concat
net.layers(index) = net.layers(end);
net.layers(index).name = 'concat';
net.layers(index+1).inputs = {'x16_ah'};
net.layers(index+2).inputs = {'x16_ah'};
net.rebuild();
net.removeLayer('concat__');
nvec = 3 * opts.num_actions * opts.num_action_history;
net.params(11).value = cat(3, net.params(11).value, zeros(1,1,nvec,3*opts.num_actions));
net.params(13).value = cat(3, net.params(13).value, zeros(1,1,nvec,3*2));

% Convert softmaxlog -> policy gradient loss
for id = 1:3
    net.layers(net.getLayerIndex(['loss' num2str(id)])).block.loss = 'softmaxlog_pg';
end

% Initialize momentum
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

% Set SGD Execution order 
exec_order = net.getLayerExecutionOrder;
exec_order_action = exec_order;
exec_order_action(end-8:2:end) = [];

% =========================================================================
% RUN TRACKING ON TRAINING VIDEOS
% =========================================================================
opts.minibatch_size = 32;

tic_train = tic;

fprintf('Starting Reinforcement Learning...\n\n');
for epoch = 1 : opts.numEpoch
    count=1;
    
    % randomly choose video/ID from database
    vid_idxs=randperm(length(all_vid_info));
    for vid_idx = vid_idxs
        vid_info=all_vid_info{vid_idx};

        % a few initializing things
        curr_img = imread(fullfile(vid_info.img_files(1).name));
        imSize = size(curr_img);
        opts.imgSize = imSize;
        num_show_actions = 20;

        % -----------------------------------------------------------------
        % Load FC6 Layer's Parameters for Multi-domain Learning
        % -----------------------------------------------------------------

        % more moving to gpu stuff
        net.move('gpu');
        
%         dummy_net = dummy_nets{count};
%         dummy_net.move('gpu');        
%         layer_name = ['v' num2str(count)];
%         fc_layer = dummy_net.getLayer(layer_name);
%         net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:fc5var,:) = ...
%             dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value;
%         net.params(net.getParamIndex(fc_layer.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value;
% 
%         dummy_score_net = dummy_score_nets{count};
%         dummy_score_net.move('gpu');
%         layer_name = ['vs' num2str(count)];
%         fc_layer_score = dummy_score_net.getLayer(layer_name);
%         net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:fc5var,:) = ...
%             dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value;
%         net.params(net.getParamIndex(fc_layer_score.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value;

        % same for net_conv and net_fc
        [net, net_conv, net_fc] = split_dagNN(net);
        net_conv.move('gpu');
        net_fc.move('gpu');
        for id = 1:3
            obj_idx = net_fc.getVarIndex(['objective' num2str(id)]);	% TODO
            pred_idx = net_fc.getVarIndex(['prediction' num2str(id)]);
            net_fc.vars(obj_idx).precious = 1;
            net_fc.vars(pred_idx).precious = 1;  
        end
        conv_feat_idx = net_conv.getVarIndex('x10');
        net_conv.vars(conv_feat_idx).precious = 1;

        % -----------------------------------------------------------------
        % Sample k video clips (between the given ground truths)
        % -----------------------------------------------------------------        
        RL_steps = opts.train.RL_steps; % alg 1 line 3
        vid_clip_starts = 1:size(vid_info.gt, 1) - RL_steps;
        vid_clip_ends = vid_clip_starts + RL_steps;
        vid_clip_ends(end) = size(vid_info.gt, 1);
        if vid_clip_starts(end) == vid_clip_ends(end)
            vid_clip_starts(end) = [];
            vid_clip_ends(end) = [];
        end
        % randomly mix up vid clip starts and ends
        randp = randperm(numel(vid_clip_starts));
        vid_clip_starts = vid_clip_starts(randp);
        vid_clip_ends = vid_clip_ends(randp);
        num_train_clips = min(opts.train.rl_num_batches, numel(vid_clip_starts));

        % -----------------------------------------------------------------
        % Play sampled video clips & calculate target value (z)
        % -----------------------------------------------------------------

        % initialize arrays
        curr_loss = 0;
        ah_all = [];
        imgs_all = []; action_labels_all = [];
        for clipIdx = 1:num_train_clips
            frameStart = vid_clip_starts(clipIdx);
            frameEnd = vid_clip_ends(clipIdx);
            curr_bboxes = vid_info.gt(frameStart,:,:);  % alg 1 line 4
            action_history = zeros(num_show_actions, 3);    % alg 1 line 5
%                 this_actions = zeros(num_show_actions, 1);
            %
            imgs = [];
            action_labels = [];
            ah_threehots = [];

            % for rest of small clip, use tracker_rl to get next action
            for frameIdx = frameStart+1:frameEnd
                curr_img = imread(vid_info.img_files(frameIdx).name);
                if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end

                [curr_bboxes, action_history, ~, ~, ims, actions, ah_threehot, ~] = ...
                    tracker_rl_mot_v3(curr_img, curr_bboxes, net_conv, net_fc, action_history, opts);    % alg 1 line 8
                ah_threehots = cat(2, ah_threehots, ah_threehot);
                imgs = cat(4, imgs, ims);
                action_labels = cat(1, action_labels, actions);
            end
            % target value (z)

            % TODO: separate by ID
            ah_all = cat(2, ah_all, ah_threehots);
            imgs_all = cat(4, imgs_all, imgs);
            action_labels_all = cat(1, action_labels_all, action_labels);                 
        end

        % -----------------------------------------------------------------
        % RL Training using Policy Gradient
        % -----------------------------------------------------------------
        num_all = size(action_labels_all, 1);     
        train_all_cnt = 0;
        train_all = [];
        batch_size = opts.minibatch_size;

        % find if num_pos or num_neg has taken majority of batch size
        % and make train_pos or train_neg
        if num_all > batch_size/2   % TODO: find out if this is important when pos/neg taken out
            % Random permutation batches (Pos)
            remain = opts.minibatch_size * num_train_clips;
            while(remain>0)
                if(train_all_cnt==0)
                    train_all_list = randperm(num_all);
                    train_all_list = train_all_list';
                end
                train_all = cat(1,train_all,...
                    train_all_list(train_all_cnt+1:min(numel(train_all_list),train_all_cnt+remain)));
                train_all_cnt = min(length(train_all_list),train_all_cnt+remain);
                train_all_cnt = mod(train_all_cnt,length(train_all_list));
                remain = opts.minibatch_size * num_train_clips-length(train_all);
            end
        end        

        % Training       
        for batchIdx = 1 : num_train_clips            
            % choose train_pos or train_neg: depends on previous portion
            % legit training portion: get inputs, outputs, and evaluate
            if ~isempty(train_all)
                all_examples = train_all((batchIdx-1)*batch_size+1:batchIdx*batch_size);
                imgs = imgs_all(:,:,:, all_examples);
                action_labels = action_labels_all(all_examples,:);   
                action_labels = gpuArray(action_labels);
                ahs = ah_all(:,all_examples);
                ahs = reshape(ahs, [1,1, size(ahs,1), size(ahs,2)]);
                inputs = {'input', gpuArray(imgs), 'label1', action_labels(:,1), ...
                    'label2', action_labels(:,2), 'label3', action_labels(:,3), ...
                    'action_history', gpuArray(ahs)};
                outputs = {'objective1', 1, 'objective2', 1, 'objective3', 1};
                net.setExecutionOrder(exec_order_action);
                target = 1;

                for id = 1:3
                    net.layers(net.getLayerIndex(['loss' num2str(id)])).block.opts = {'target', target / batch_size}; % alg 1 line 11?
                end

                net.eval(inputs,outputs);  % alg 1 line 10

                [state] = accumulate_gradients_dagnn(state, net, opts.train, batch_size);

                loss_id = 0;
                for id = 1:3
                    loss_id = loss_id + sum(gather(net.getVar(['objective' num2str(id)]).value)) / batch_size;
                end
                curr_loss = curr_loss + loss_id / 3;
            end
        end

        % Print objective (loss)
        curr_loss = curr_loss / num_train_clips;
        fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d, loss: %d \n', ...
            toc(tic_train), epoch, opts.numEpoch, count, length(all_vid_info), curr_loss);

        % -----------------------------------------------------------------
        % Save current parameters
        % -----------------------------------------------------------------

%         % move dummy net and dummy score net back for next layer
%         dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value = net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:fc5var,:);
%         dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value = net.params(net.getParamIndex(fc_layer.params{2})).value;
%         dummy_net.move('cpu');
%         dummy_nets{count} = dummy_net;  % alg 1 line 12
%         dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value = net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:fc5var,:);
%         dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value = net.params(net.getParamIndex(fc_layer_score.params{2})).value;
%         dummy_score_net.move('cpu');
%         dummy_score_nets{count} = dummy_score_net;  

        count = count+1;
    end
end

% -------------------------------------------------------------------------
% FINALIZE TRAINING
% -------------------------------------------------------------------------
net.move('cpu');
% net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:fc5var,:) = 0.01 * randn(1,1,fc5var,opts.num_actions,'single');
% net.params(net.getParamIndex(fc_layer.params{2})).value = zeros(1, opts.num_actions, 'single');
% net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:fc5var,:) = 0.01 * randn(1,1,fc5var,2,'single');
% net.params(net.getParamIndex(fc_layer_score.params{2})).value = zeros(1, 2, 'single');

% softmaxlossBlock = dagnn.Loss('loss', 'softmaxlog');
% net.addLayer('loss_score',softmaxlossBlock,{'prediction_score', 'label_score'},{'objective_score'});

% save(sprintf('./models/net_dummy_%s',tail), 'dummy_nets', 'dummy_score_nets', '-v7.3');
