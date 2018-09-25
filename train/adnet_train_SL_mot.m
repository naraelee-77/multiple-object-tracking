function [net, all_vid_info] = adnet_train_SL_mot(opts)
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
train_db_path = './models/train_db_mot.mat';
if ~exist(train_db_path, 'file')
    fprintf('Num videos: %d\n', opts.num_videos);
    train_db={};
    all_vid_info={};
    for vid_idx = 1 : opts.num_videos
        vid_name = vid_names(vid_idx).name;
        pre_vid_info = get_pre_vid_infos_mot(vid_path, vid_name);
        id_combos = combnk(pre_vid_info.ids, 3);
        combo_order=randperm(size(id_combos,1));
        id_combos=id_combos(combo_order,:);
        
        count=1;
        for num = 1:size(id_combos,1)
            rand_ids = id_combos(num,:);
            [vid_info, success] = get_vid_infos_mot(pre_vid_info, rand_ids);
            if ~success
                continue;
            end

            train_db{end+1} = get_train_dbs_mot(vid_info, opts);
            all_vid_info{end+1} = vid_info;
            fprintf('Video %d\tIDs [%s]\tcount %d\n', vid_idx, sprintf('%d ', rand_ids), count);
            
            if count >= 25
                break;
            end
            count=count+1;
        end
    end
    save( train_db_path, 'train_db', 'all_vid_info', '-v7.3');
else
    load( train_db_path, 'train_db', 'all_vid_info');
end
opts.num_videos=numel(train_db);

if exist('./models/net_sl_mot.mat', 'file')
    fprintf('Supervised Learning completed.\n');
    load('./models/net_sl_mot.mat', 'net')
    return;
end

% ========================================================
% Init model
% ========================================================

% initialize model with imagenet-vgg-m-conv1-3.mat, add fc layers
[net, dummy_nets, dummy_score_nets] = init_model_mot('./models/net_init_mot.mat', opts);

%
tic_train = tic;
train_cost = zeros(opts.numEpoch, numel(train_db));
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

exec_order = net.getLayerExecutionOrder;
exec_order_action = exec_order;
exec_order_action(end-10:2:end) = [];
exec_order_score = exec_order;
exec_order_score(end-11:2:end-1) = [];

opts.frameBatch = 5;
opts.minibatch_size = 128;

fprintf('Starting Supervised Learning...\n\n');
for epoch = 1 : opts.numEpoch
    for i = 1 : numel(train_db)
        % move to gpu and get dummy net layer, fc layer
        net.move('gpu');
        dummy_net = dummy_nets{i};
        dummy_net.move('gpu');
        dummy_score_net = dummy_score_nets{i};
        dummy_score_net.move('gpu');
        
        for id = 1:3
            layer_name = sprintf('v%d_id%d', i, id);  
            fc_layer{id} = dummy_net.getLayer(layer_name);    
            net.params(net.getParamIndex(fc_layer{id}.params{1})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer{id}.params{1})).value;
            net.params(net.getParamIndex(fc_layer{id}.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer{id}.params{2})).value;
            
            layer_name = sprintf('vs%d_id%d', i, id); 
            fc_layer_score{id} = dummy_score_net.getLayer(layer_name);    
            net.params(net.getParamIndex(fc_layer_score{id}.params{1})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score{id}.params{1})).value;
            net.params(net.getParamIndex(fc_layer_score{id}.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score{id}.params{2})).value;
        end
        
        % same as above for dummy score nets, fc layer score
        % shuffle frames during training
        frame_batch_shuffled = randperm(numel(train_db{i}));

        for j = 1 : min (numel(train_db{i}), opts.frameBatch)
            
            % get current image, data from train_db, check if grayscale
            train_img = imread(train_db{i}(frame_batch_shuffled(j)).img_path);
            bboxes = train_db{i}(frame_batch_shuffled(j)).bboxes;
            action_labels = train_db{i}(frame_batch_shuffled(j)).labels;
            score_labels = train_db{i}(frame_batch_shuffled(j)).score_labels;
            if(size(train_img,3)==1), train_img = cat(3,train_img,train_img,train_img); end
            
            imgs = cell(1,3);
            labels = cell(1,3);
            for id = 1:3
            % look at just the portion of the image from pos and neg examples
                imgs{id} = get_extract_regions(train_img, bboxes(:,:,id));
                imgs{id} = gpuArray(imgs{id});
                inds = find(action_labels(:,:,id));
                [labels{id},~] = ind2sub(size(action_labels(:,:,id)), inds);
                labels{id} = gpuArray(labels{id});
            end
            
            % training batches from one image
            % for action network
            net.setExecutionOrder(exec_order_action);
            num_data = numel(labels{1});
            num_batches = num_data / opts.minibatch_size;
            batch_shuffled = randperm(num_data);
            for k = 1 : num_batches + 1
                % get random batch
                batch_start =  (k-1)*opts.minibatch_size+1 ;
                batch_end = min(num_data, k*opts.minibatch_size) ;
                curr_batch_idx = batch_shuffled(batch_start:batch_end);
                
                % evaluate performance for this single action chosen.
                % calculate loss and cost
                
                inputs = {
                    'input_id1', imgs{1}(:,:,:,curr_batch_idx), ...
                    'input_id2', imgs{2}(:,:,:,curr_batch_idx), ...
                    'input_id3', imgs{3}(:,:,:,curr_batch_idx), ...
                    'label_id1', labels{1}(curr_batch_idx), ...
                    'label_id2', labels{2}(curr_batch_idx), ...
                    'label_id3', labels{3}(curr_batch_idx)
                };
                outputs = {'objective_id1', 1, 'objective_id2', 1, 'objective_id3', 1};        
                net.eval(inputs, outputs) ;
                [state] = accumulate_gradients_dagnn(state, net, opts.train, opts.minibatch_size);
                curr_loss_id1 = gather(net.getVar('objective_id1').value);  
                curr_loss_id2 = gather(net.getVar('objective_id2').value);  
                curr_loss_id3 = gather(net.getVar('objective_id3').value);  
                avg_curr_loss = (sum(curr_loss_id1) + sum(curr_loss_id2) + sum(curr_loss_id3))/3;
                train_cost(epoch,i) = train_cost(epoch,i) + avg_curr_loss;
            end
            
            % same as above for score network
            net.setExecutionOrder(exec_order_score);
            num_data = size(score_labels,1);
            num_batches = num_data / opts.minibatch_size;
            batch_shuffled = randperm(num_data);
            for k = 1 : num_batches + 1
                batch_start =  (k-1)*opts.minibatch_size+1 ;
                batch_end = min(num_data, k*opts.minibatch_size) ;
                curr_batch_idx = batch_shuffled(batch_start:batch_end);
                
                inputs = {
                    'input_id1', imgs{1}(:,:,:,curr_batch_idx), ...
                    'input_id2', imgs{2}(:,:,:,curr_batch_idx), ...
                    'input_id3', imgs{3}(:,:,:,curr_batch_idx), ...
                    'label_score_id1', score_labels(curr_batch_idx,1), ...
                    'label_score_id2', score_labels(curr_batch_idx,2), ...
                    'label_score_id3', score_labels(curr_batch_idx,3)
                };
                outputs = {'objective_score_id1', 1, 'objective_score_id2', 1, 'objective_score_id3', 1};        
                net.eval(inputs, outputs) ;
                [state] = accumulate_gradients_dagnn(state, net, opts.train, opts.minibatch_size);
                curr_loss_id1 = gather(net.getVar('objective_score_id1').value);  
                curr_loss_id2 = gather(net.getVar('objective_score_id2').value);  
                curr_loss_id3 = gather(net.getVar('objective_score_id3').value);  
                avg_curr_loss = (sum(curr_loss_id1) + sum(curr_loss_id2) + sum(curr_loss_id3))/3;
                train_cost(epoch,i) = train_cost(epoch,i) + avg_curr_loss;
            end       
        end
        % print info
        train_cost(epoch,i) = train_cost(epoch,i) / opts.minibatch_size;
        fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d, cost: %f \n', ... 
                toc(tic_train), epoch, opts.numEpoch, i, numel(train_db), train_cost(epoch,i));
            
        % move back to cpu and save dummy nets and dummy score nets for next layer
        for id = 1:3
            dummy_net.params(dummy_net.getParamIndex(fc_layer{id}.params{1})).value = net.params(net.getParamIndex(fc_layer{id}.params{1})).value;
            dummy_net.params(dummy_net.getParamIndex(fc_layer{id}.params{2})).value = net.params(net.getParamIndex(fc_layer{id}.params{2})).value;

            dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score{id}.params{1})).value = net.params(net.getParamIndex(fc_layer_score{id}.params{1})).value;
            dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score{id}.params{2})).value = net.params(net.getParamIndex(fc_layer_score{id}.params{2})).value;
        end

        dummy_net.move('cpu');
        dummy_nets{i} = dummy_net;
        dummy_score_net.move('cpu');
        dummy_score_nets{i} = dummy_score_net;
    end
end

% FINALIZE TRAINING
net.move('cpu');
for id = 1:3
    net.params(net.getParamIndex(fc_layer{id}.params{1})).value = 0.01 * randn(3,1,512,opts.num_actions,'single');
    net.params(net.getParamIndex(fc_layer{id}.params{2})).value = zeros(1, opts.num_actions, 'single');
    net.params(net.getParamIndex(fc_layer_score{id}.params{1})).value = 0.01 * randn(3,1,512,2,'single');
    net.params(net.getParamIndex(fc_layer_score{id}.params{2})).value = zeros(1, 2, 'single');
end

save('./models/net_dummy_mot.mat', 'dummy_nets', 'dummy_score_nets', '-v7.3');
