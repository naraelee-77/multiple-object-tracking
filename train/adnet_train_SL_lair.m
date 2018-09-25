function [net, all_vid_info] = adnet_train_SL_lair(opts,x,tail)
% ADNET_TRAIN_SL Train the ADNet with supervised learning
% 
% Sangdoo Yun, 2017.

% ========================================================
% Set train DB
% ========================================================
train_videos = get_train_videos(opts);
opts.num_videos = numel(train_videos.video_names);


% ========================================================
% Train - supervised learning
% ========================================================

% Constructing database: For each video and for each ID, make database of
% vid_info, which contains video name, image files, gt bboxes.
% Also make train_db, which also has pos and neg examples, action labels and score labels
fprintf('Construct database. \n');
train_db_path = sprintf('./models/train_db_%s.mat',tail);
% fprintf('%s\n', train_db_path);
if ~exist(train_db_path, 'file')
    fprintf('Num videos: %d\n', opts.num_videos);
    count=1;
    for vid_idx = 1 : opts.num_videos
        vid_name = train_videos.video_names{vid_idx};
        vid_path = train_videos.video_paths{vid_idx};
        pre_vid_info = get_pre_vid_infos_lair(vid_path, vid_name);
        id_idxs=randperm(pre_vid_info.nids);
        i=1;
        for id_idx = id_idxs
            vid_info = get_vid_infos_lair(pre_vid_info, id_idx);
            try
                lastwarn('');
                for fr = 1:vid_info.nframes
                    img = imread(vid_info.img_files(fr).name);
                end
                [warnMsg, ~]=lastwarn;
                if ~isempty(warnMsg)
                    fprintf('Skipping Video %d, ID %d\n', vid_idx, pre_vid_info.ids(id_idx));
                    continue;
                end
                
                train_db{count} = get_train_dbs(vid_info, opts);
                all_vid_info{count} = vid_info;
                fprintf('Video %d, ID %d, total %d\n', vid_idx, pre_vid_info.ids(id_idx), count);
                count=count+1;
            catch
                fprintf('Skipping Video %d, ID %d\n', vid_idx, pre_vid_info.ids(id_idx));
                continue;
            end
            if mod(i,x)==0
                break;
            end
            i=i+1;
        end
    end
    save( train_db_path, 'train_db', 'all_vid_info', '-v7.3');
else
    load( train_db_path, 'train_db', 'all_vid_info');
end
opts.num_videos=numel(train_db);

if exist(sprintf('./models/net_sl_%s.mat', tail), 'file')
    fprintf('Supervised Learning completed.\n');
    load(sprintf('./models/net_sl_%s.mat', tail), 'net')
    return;
end

% ========================================================
% Init model
% ========================================================

% initialize model with imagenet-vgg-m-conv1-3.mat, add fc layers
[net, dummy_nets, dummy_score_nets] = init_model(sprintf('./models/net_init_%s.mat',tail), opts);
net.vars(net.getVarIndex('objective')).precious = 1;
net.vars(net.getVarIndex('objective_score')).precious = 1;

%
tic_train = tic;
train_cost = zeros(opts.numEpoch, numel(train_db));
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

exec_order = net.getLayerExecutionOrder;
exec_order_action = exec_order;
exec_order_action(end-2:2:end) = [];
exec_order_score = exec_order;
exec_order_score(end-3:2:end-1) = [];

opts.frameBatch = 5;
opts.minibatch_size = 128;

fprintf('Starting Supervised Learning...\n\n');
for epoch = 1 : opts.numEpoch
    for i = 1 : numel(train_db)
        % move to gpu and get dummy net layer, fc layer
        net.move('gpu');
        dummy_net = dummy_nets{i};
        dummy_net.move('gpu');
        layer_name = ['v' num2str(i)];  
        fc_layer = dummy_net.getLayer(layer_name);    
        net.params(net.getParamIndex(fc_layer.params{1})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value;
        net.params(net.getParamIndex(fc_layer.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value;
        
        % same as above for dummy score nets, fc layer score
        dummy_score_net = dummy_score_nets{i};
        dummy_score_net.move('gpu');
        layer_name = ['vs' num2str(i)];  
        fc_layer_score = dummy_score_net.getLayer(layer_name);    
        net.params(net.getParamIndex(fc_layer_score.params{1})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value;
        net.params(net.getParamIndex(fc_layer_score.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value;
        
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
            imgs = get_extract_regions(train_img, bboxes);
            imgs = gpuArray(imgs);
            
            % from ground truth, get best action (translation/scale/etc)
            inds = find(action_labels);
            [labels,~] = ind2sub(size(action_labels), inds);
            labels = gpuArray(labels);
            
            % training batches from one image
            % for action network
            net.setExecutionOrder(exec_order_action);
            num_data = numel(labels);
            num_batches = num_data / opts.minibatch_size;
            batch_shuffled = randperm(num_data);
            for k = 1 : num_batches + 1
                % get random batch
                batch_start =  (k-1)*opts.minibatch_size+1 ;
                batch_end = min(num_data, k*opts.minibatch_size) ;
                curr_batch_idx = batch_shuffled(batch_start:batch_end);
                
                % evaluate performance for this single action chosen.
                % calculate loss and cost
                inputs = {'input', imgs(:,:,:,curr_batch_idx), 'label', labels(curr_batch_idx)};
                outputs = {'objective', 1};        
                net.eval(inputs, outputs) ;
                [state] = accumulate_gradients_dagnn(state, net, opts.train, opts.minibatch_size);
                curr_loss = gather(net.getVar('objective').value);  
                train_cost(epoch,i) = train_cost(epoch,i) + sum(curr_loss);
            end
            
            % same as above for score network
            net.setExecutionOrder(exec_order_score);
            num_data = numel(score_labels);
            num_batches = num_data / opts.minibatch_size;
            batch_shuffled = randperm(num_data);
            for k = 1 : num_batches + 1
                batch_start =  (k-1)*opts.minibatch_size+1 ;
                batch_end = min(num_data, k*opts.minibatch_size) ;
                curr_batch_idx = batch_shuffled(batch_start:batch_end);
                
                inputs = {'input', imgs(:,:,:,curr_batch_idx), 'label_score', gpuArray( score_labels(curr_batch_idx))};
                outputs = {'objective_score', 1};        
                net.eval(inputs, outputs) ;
                [state] = accumulate_gradients_dagnn(state, net, opts.train, opts.minibatch_size);
                curr_loss = gather(net.getVar('objective_score').value);
                train_cost(epoch,i) = train_cost(epoch,i) + sum(curr_loss);
            end       
        end
        % print info
        train_cost(epoch,i) = train_cost(epoch,i) / opts.minibatch_size;
        fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d, cost: %f \n', ... 
                toc(tic_train), epoch, opts.numEpoch, i, numel(train_db), train_cost(epoch,i));
            
        % move back to cpu and save dummy nets and dummy score nets for next layer
        dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value = net.params(net.getParamIndex(fc_layer.params{1})).value;
        dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value = net.params(net.getParamIndex(fc_layer.params{2})).value;
        dummy_net.move('cpu');
        dummy_nets{i} = dummy_net;
        
        dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value = net.params(net.getParamIndex(fc_layer_score.params{1})).value;
        dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value = net.params(net.getParamIndex(fc_layer_score.params{2})).value;
        dummy_score_net.move('cpu');
        dummy_score_nets{i} = dummy_score_net;
    end
end

% FINALIZE TRAINING
net.move('cpu');
net.params(net.getParamIndex(fc_layer.params{1})).value = 0.01 * randn(1,1,512,opts.num_actions,'single');
net.params(net.getParamIndex(fc_layer.params{2})).value = zeros(1, opts.num_actions, 'single');
net.params(net.getParamIndex(fc_layer_score.params{1})).value = 0.01 * randn(1,1,512,2,'single');
net.params(net.getParamIndex(fc_layer_score.params{2})).value = zeros(1, 2, 'single');

save(sprintf('./models/net_dummy_%s.mat',tail), 'dummy_nets', 'dummy_score_nets', '-v7.3');
