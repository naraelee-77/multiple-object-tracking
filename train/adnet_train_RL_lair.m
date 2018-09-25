function net = adnet_train_RL_lair(net, all_vid_info, opts, tail)
% ADNET_TRAIN_RL Train the ADNet with reinforcement learning
% 
% Sangdoo Yun, 2017.

% model from SL
load(sprintf('./models/net_dummy_%s.mat',tail)); % alg 1 line 1

% change layers around
net.addLayer('concat_',dagnn.Concat, {'x16', 'action_history'}, 'x_16_ah');
net.layers(end-3:end-1) = net.layers(end-4:end-2); % make room for concat
net.layers(17) = net.layers(end);
net.layers(17).name = 'concat';
net.layers(18).inputs = {'x_16_ah'};
net.layers(19).inputs = {'x_16_ah'};
net.removeLayer('concat_');
net.rebuild();
nvec = opts.num_actions * opts.num_action_history;
net.params(11).value = cat(3, net.params(11).value, zeros(1,1,nvec,11));
net.params(13).value = cat(3, net.params(13).value, zeros(1,1,nvec,2));

% Convert softmaxlog -> policy gradient loss
net.layers(net.getLayerIndex('loss')).block.loss = 'softmaxlog_pg';

% Initialize momentum
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

% Set SGD Execution order 
exec_order = net.getLayerExecutionOrder;
exec_order_action = exec_order;
exec_order_action(end-1:2:end) = [];

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
%         try
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
            dummy_net = dummy_nets{count};
            dummy_net.move('gpu');        
            layer_name = ['v' num2str(count)];
            fc_layer = dummy_net.getLayer(layer_name);
            net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:) = ...
                dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value;
            net.params(net.getParamIndex(fc_layer.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value;

            dummy_score_net = dummy_score_nets{count};
            dummy_score_net.move('gpu');
            layer_name = ['vs' num2str(count)];
            fc_layer_score = dummy_score_net.getLayer(layer_name);
            net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:) = ...
                dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value;
            net.params(net.getParamIndex(fc_layer_score.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value;

            % same for net_conv and net_fc
            [net, net_conv, net_fc] = split_dagNN(net);
            net_conv.move('gpu');
            net_fc.move('gpu');
            obj_idx = net_fc.getVarIndex('objective');        
            pred_idx = net_fc.getVarIndex('prediction');        
            conv_feat_idx = net_conv.getVarIndex('x10');
            net_fc.vars(obj_idx).precious = 1;
            net_fc.vars(pred_idx).precious = 1;
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
            ah_pos = []; ah_neg = [];
            imgs_pos = []; action_labels_pos = [];
            imgs_neg = []; action_labels_neg = [];
            for clipIdx = 1:num_train_clips
                frameStart = vid_clip_starts(clipIdx);
                frameEnd = vid_clip_ends(clipIdx);
                curr_bbox = vid_info.gt(frameStart,:);  % alg 1 line 4
                action_history = zeros(num_show_actions, 1);    % alg 1 line 5
%                 this_actions = zeros(num_show_actions, 1);
                %
                imgs = [];
                action_labels = [];
                ah_onehots = [];
                
                % for rest of small clip, use tracker_rl to get next action
                for frameIdx = frameStart+1:frameEnd
                    curr_img = imread(vid_info.img_files(frameIdx).name);
                    if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end

                    curr_gt_bbox = vid_info.gt(frameIdx,:);
                    [curr_bbox, action_history, ~, ~, ims, actions, ah_onehot, ~] = ...
                        tracker_rl(curr_img, curr_bbox, net_conv, net_fc, action_history, opts);    % alg 1 line 8
                    ah_onehots = cat(2, ah_onehots, ah_onehot);
                    imgs = cat(4, imgs, ims);
                    action_labels = cat(1, action_labels, actions);
                end
                % target value (z)
                
                % pos and neg results. > or <= 0.7
                if overlap_ratio(curr_gt_bbox, curr_bbox) > 0.7
                    ah_pos = cat(2, ah_pos, ah_onehots);
                    imgs_pos = cat(4, imgs_pos, imgs);
                    action_labels_pos = cat(1, action_labels_pos, action_labels);
                else
                    ah_neg = cat(2, ah_neg, ah_onehots);
                    imgs_neg = cat(4, imgs_neg, imgs);
                    action_labels_neg = cat(1, action_labels_neg, action_labels);
                end                      
            end

            % -----------------------------------------------------------------
            % RL Training using Policy Gradient
            % -----------------------------------------------------------------
            num_pos = size(action_labels_pos, 1);
            num_neg = size(action_labels_neg, 1);        
            train_pos_cnt = 0;
            train_pos = [];
            train_neg_cnt = 0;
            train_neg = [];
            batch_size = opts.minibatch_size;
            
            % find if num_pos or num_neg has taken majority of batch size
            % and make train_pos or train_neg
            if num_pos > batch_size/2
                % Random permutation batches (Pos)
                remain = opts.minibatch_size * num_train_clips;
                while(remain>0)
                    if(train_pos_cnt==0)
                        train_pos_list = randperm(num_pos);
                        train_pos_list = train_pos_list';
                    end
                    train_pos = cat(1,train_pos,...
                        train_pos_list(train_pos_cnt+1:min(numel(train_pos_list),train_pos_cnt+remain)));
                    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
                    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
                    remain = opts.minibatch_size * num_train_clips-length(train_pos);
                end
            end        
            if num_neg > batch_size/2
                % Random permutation batches (Neg)
                remain = opts.minibatch_size * num_train_clips;

                while(remain>0)
                    if(train_neg_cnt==0)
                        train_neg_list = randperm(num_neg);
                        train_neg_list = train_neg_list';
                    end
                    train_neg = cat(1,train_neg,...
                        train_neg_list(train_neg_cnt+1:min(numel(train_neg_list),train_neg_cnt+remain)));
                    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
                    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
                    remain = opts.minibatch_size * num_train_clips-length(train_neg);
                end
            end

            % Training       
            for batchIdx = 1 : num_train_clips            
                % choose train_pos or train_neg: depends on previous portion
                % legit training portion: get inputs, outputs, and evaluate
                if ~isempty(train_pos)
                    pos_examples = train_pos((batchIdx-1)*batch_size+1:batchIdx*batch_size);
                    imgs = imgs_pos(:,:,:, pos_examples);
                    action_labels = action_labels_pos(pos_examples);   
                    ahs = ah_pos(:,pos_examples);
                    ahs = reshape(ahs, [1,1, size(ahs,1), size(ahs,2)]);
%                     whos imgs action_labels ahs
                    inputs = {'input', gpuArray(imgs), 'label', gpuArray(action_labels), 'action_history', gpuArray(ahs)};
%                     outputs={'x1', gpuArray(zeros(53,53,96,32))};
                    outputs = {'objective', 1};
                    net.setExecutionOrder(exec_order_action);
                    target = 1;
                    net.layers(net.getLayerIndex('loss')).block.opts = {'target', target / batch_size}; % alg 1 line 11?
                    net.eval(inputs,outputs);  % alg 1 line 10
                    
%                     disp(net.getVarSizes());

                    for var=1:numel(net.vars)
                        disp(net.vars(var));
                    end
                    
                    [state] = accumulate_gradients_dagnn(state, net, opts.train, batch_size);
                    curr_loss = curr_loss + gather(net.getVar('objective').value) / batch_size;
                end
                if ~isempty(train_neg)
                    neg_examples = train_neg((batchIdx-1)*batch_size+1:batchIdx*batch_size);
                    imgs = imgs_neg(:,:,:, neg_examples);
                    action_labels = action_labels_neg(neg_examples);
                    ahs = ah_neg(:,neg_examples);
                    ahs = reshape(ahs, [1,1, size(ahs,1), size(ahs,2)]);
                    inputs = {'input', gpuArray(imgs), 'label', gpuArray(action_labels), 'action_history', gpuArray(ahs)};
                    outputs = {'objective', 1};
                    net.setExecutionOrder(exec_order_action);
                    target = -1;
                    net.layers(net.getLayerIndex('loss')).block.opts = {'target', target / batch_size}; % alg 1 line 11?
                    net.eval(inputs,outputs);  % alg 1 line 10
                    
                    for var=1:numel(net.vars)
                        disp(net.vars(var));
                    end
                    
                    [state] = accumulate_gradients_dagnn(state, net, opts.train, batch_size);
                    curr_loss = curr_loss + gather(net.getVar('objective').value) / batch_size;
                end
            end

            % Print objective (loss)
            curr_loss = curr_loss / num_train_clips;
            fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d, loss: %d \n', ...
                toc(tic_train), epoch, opts.numEpoch, count, length(all_vid_info), curr_loss);

            % -----------------------------------------------------------------
            % Save current parameters
            % -----------------------------------------------------------------
            
            % move dummy net and dummy score net back for next layer
            dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value = net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:);
            dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value = net.params(net.getParamIndex(fc_layer.params{2})).value;
            dummy_net.move('cpu');
            dummy_nets{count} = dummy_net;  % alg 1 line 12
            dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value = net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:);
            dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value = net.params(net.getParamIndex(fc_layer_score.params{2})).value;
            dummy_score_net.move('cpu');
            dummy_score_nets{count} = dummy_score_net;  
%         catch
%             fprintf('Something went wrong with vid %d\n', count);
%         end

        count = count+1;
    end
end

% -------------------------------------------------------------------------
% FINALIZE TRAINING
% -------------------------------------------------------------------------
net.move('cpu');
net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:) = 0.01 * randn(1,1,512,opts.num_actions,'single');
net.params(net.getParamIndex(fc_layer.params{2})).value = zeros(1, opts.num_actions, 'single');
net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:) = 0.01 * randn(1,1,512,2,'single');
net.params(net.getParamIndex(fc_layer_score.params{2})).value = zeros(1, 2, 'single');

softmaxlossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('loss_score',softmaxlossBlock,{'prediction_score', 'label_score'},{'objective_score'});

