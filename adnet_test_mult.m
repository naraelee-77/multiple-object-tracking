function [bboxes, t_sum, precisions] = adnet_test_mult(vid_path, model_name, idnums, interval)
% ADNET_DEMO Run tracking on a sequence
%
% Sangdoo Yun, 2017.

if nargin < 4
    interval=1;
end

addpath('test/');
addpath(genpath('utils/'));

init_settings;

run(matconvnet_path);

load(fullfile('models', strcat(model_name,'.mat')));

opts.visualize = true;
opts.printscreen = true;

fprintf('Testing sequences in %s... \n', vid_path);
t_sum = 0;

% ============================
% Load video sequences
% ============================

gt_all = importdata(fullfile(vid_path, 'gt.txt'));
gt_all = gt_all(:,1:6);
gt_new = cell(1, length(idnums));

% have to add interval n~=1 support
% for each ID, find beginning and ending frames to find intersection of all
begin_frame=0;
end_frame=0;
curr_frame=1;
count_mems=0;
lines = zeros(1, length(idnums));
for line = 1:length(gt_all)
    if gt_all(line,1) ~= curr_frame
        if count_mems == length(idnums)
            for i = 1:length(idnums)
                gt_new{i} = cat(1, gt_new{i}, gt_all(lines(i),:));
            end
            if begin_frame == 0
                begin_frame = curr_frame;
            end
            end_frame = curr_frame;
        end
        curr_frame = curr_frame + 1;
        count_mems = 0;
        lines = zeros(1, length(idnums));
    end
    if ismember(gt_all(line,2), idnums)
        count_mems = count_mems+1;
        lines(gt_all(line,2)==idnums) = line;
    end
end

if end_frame == 0
    fprintf('All IDs are never in the same frame. Exiting.\n');
    return;
end

vid_info.gt = gt_new;
vid_info.img_files = [];
count=1;
for frame = begin_frame:interval:end_frame
    vid_info.img_files(count).name = fullfile(vid_path, 'img', sprintf('%04d.jpg', frame));
%     for i = 1:length(idnums)
%         gt_new{i}=cat(1,gt_new{i},gt_all(frame,:));
%     end
    count=count+1;
end

vid_info.nframes = numel(vid_info.img_files);

folder_name=sprintf('%s_ids%sn%01d', model_name, sprintf('%01d_', idnums(:)), interval);
if ~exist(fullfile(vid_path, 'img_bbox_folders', folder_name), 'dir')
    mkdir(fullfile(vid_path, 'img_bbox_folders', folder_name));
end

% init containers
bboxes = zeros([3, size(vid_info.gt{1})]);
total_pos_data = cell(length(idnums));
total_neg_data = cell(length(idnums));
total_pos_action_labels = cell(length(idnums));
total_pos_examples = cell(length(idnums));
total_neg_examples = cell(length(idnums));

for i = 1:length(idnums)
    total_pos_data{i} = cell(1,1,1,vid_info.nframes);
    total_neg_data{i} = cell(1,1,1,vid_info.nframes);
    total_pos_action_labels{i} = cell(1,vid_info.nframes);
    total_pos_examples{i} = cell(1,vid_info.nframes);
    total_neg_examples{i} = cell(1,vid_info.nframes);
end

% init model networks
net.layers(net.getLayerIndex('loss')).block.loss = 'softmaxlog';
[net, net_conv, net_fc] = split_dagNN(net);

net_fc.params(net_fc.getParamIndex('fc6_1b')).value = ...
    gpuArray(ones(size(net_fc.params(net_fc.getParamIndex('fc6_1b')).value), 'single') * 0.01);

% set params learning rate
for p = 1 : numel(net_fc.params)
    if mod(p, 2) == 1
        net_fc.params(p).learningRate = 10;
    else
        net_fc.params(p).learningRate = 20;
    end
end
net_fc.params(net_fc.getParamIndex('fc6_1f')).learningRate = 20;
net_fc.params(net_fc.getParamIndex('fc6_1b')).learningRate = 40;

obj_idx = net_fc.getVarIndex('objective');
obj_score_idx = net_fc.getVarIndex('objective_score');
pred_idx = net_fc.getVarIndex('prediction');
pred_score_idx = net_fc.getVarIndex('prediction_score');
conv_feat_idx = net_conv.getVarIndex('x10');
net_fc.vars(obj_idx).precious = 1;
net_fc.vars(obj_score_idx).precious = 1;
net_fc.vars(pred_idx).precious = 1;
net_fc.vars(pred_score_idx).precious = 1;
net_conv.vars(conv_feat_idx).precious = 1;

% move all nets to gpu
net.move('gpu');
net_conv.move('gpu');
net_fc.move('gpu');

% ============================
% RUN TRACKING
% ============================
cont_negatives = 0;
if opts.visualize == true
    res_fig = figure(1);
    set(res_fig,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
end

frame_window = cell(length(idnums));
curr_bbox = cell(length(idnums));
for frameIdx = 1 : vid_info.nframes
    
    for i = 1:length(idnums)
    
        if frameIdx == 1

            % ============================
            % LOAD & FINETUNE FC NETWORKS
            % ============================
            % initialize bbox with ground truth
            curr_bbox{i} = vid_info.gt{i}(1,3:6);
            curr_img = imread(fullfile(vid_info.img_files(1).name));
            if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end

            init_tic = tic;

            imSize = size(curr_img);
            opts.imgSize = imSize;
            action_history_oh_zeros = zeros(opts.num_actions*opts.num_action_history, 1);
            action_history_oh = action_history_oh_zeros;
            frame_tic = tic;
            % generate samples
            pos_examples = single(gen_samples('gaussian', curr_bbox{i}, opts.nPos_init*2, opts, opts.finetune_trans, opts.finetune_scale_factor));
            r = overlap_ratio(pos_examples,curr_bbox{i});
            pos_examples = pos_examples(r>opts.posThre_init,:);
            pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);
            neg_examples = [gen_samples('uniform', curr_bbox{i}, opts.nNeg_init, opts, 1, 10);...
                gen_samples('whole', curr_bbox{i}, opts.nNeg_init, opts)];
            r = overlap_ratio(neg_examples,curr_bbox{i});
            neg_examples = single(neg_examples(r<opts.negThre_init,:));
            neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);
            examples = [pos_examples; neg_examples];
            pos_idx = 1:size(pos_examples,1);
            neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
            net_conv.mode = 'test';
            feat_conv = get_conv_feature(net_conv, curr_img, examples, opts);
            pos_data = feat_conv(:,:,:,pos_idx);
            neg_data = feat_conv(:,:,:,neg_idx);
            % get action labels
            pos_action_labels = gen_action_labels(opts.num_actions, opts, pos_examples, curr_bbox{i});
            opts.maxiter = opts.finetune_iters;
            opts.learningRate = 0.0003;
            net_fc.mode = 'test';

            % configure net_fc_noah
            net_fc_noah = copy(net_fc);
            net_fc_noah.removeLayer('concat');
            net_fc_noah.layers(7).inputs = 'x16';
            net_fc_noah.layers(8).inputs = 'x16';
            net_fc_noah.params(5).value(:,:,512+1:end,:) = [];
            net_fc_noah.params(7).value(:,:,512+1:end,:) = [];
            net_fc_noah.rebuild();
            net_fc_noah.move('gpu');
            net_fc.move('gpu');

            % finetuning
            [net_fc_noah, ~] =  train_fc_finetune_hem(net_fc_noah, opts, ...
                pos_data, neg_data, pos_action_labels);
            for fci = 1 : 8
                if fci == 5 || fci == 7
                    net_fc.params(fci).value(:,:,1:512,:) = net_fc_noah.params(fci).value;
                else
                    net_fc.params(fci).value = net_fc_noah.params(fci).value;
                end
            end

            % set data to save
            total_pos_data{i}{1} = pos_data;
            total_neg_data{i}{1} = neg_data;
            total_pos_action_labels{i}{1} = pos_action_labels;
            total_pos_examples{i}{1} = pos_examples';
            total_neg_examples{i}{1} = neg_examples';

            frame_window{i} = [frame_window{i}, frameIdx];
            is_negative = false;

            if opts.printscreen == 1
                fprintf('initialize: %f sec.\n', toc(init_tic));
                fprintf('===================\n');
            end

            action_history = zeros(opts.num_show_actions, 1);
            this_actions = zeros(opts.num_show_actions, 1);
        else
            % Read image
            curr_img = imread(fullfile(vid_info.img_files(frameIdx).name));
            if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end
            curr_bbox_old = curr_bbox{i};
            frame_tic = tic;
            move_counter = 1;
            target_score = 0;

            % set previous step's data from last time's current data
            % set cap on num actions
            num_action_step_max = 20;
            bb_step = zeros(num_action_step_max, 4);
            score_step = zeros(num_action_step_max, 1);
            is_negative = false;
            prev_score = -inf;
            this_actions = zeros(opts.num_show_actions,1);
            action_history_oh_old = action_history_oh;
            
            % keep taking actions until reach stop action or total 20 actions
            while (move_counter <= num_action_step_max)
                bb_step(move_counter,:) = curr_bbox{i};
                score_step(move_counter) = prev_score;

                net_conv.mode = 'test';
                curr_feat_conv = get_conv_feature(net_conv, curr_img, curr_bbox{i}, opts);
                curr_feat_conv = gpuArray(curr_feat_conv);
                action_history_oh = get_action_history_onehot(action_history(1:opts.num_action_history), opts);
    %             action_history_oh = action_history_oh_zeros;
                action_history_oh_reshape = reshape(action_history_oh, [1,1,numel(action_history_oh),1]);
                net_fc.mode = 'test';
                net_fc.reset;
                labels = zeros(1);
                
                % evaluate from these inputs
                inputs = {'x10', curr_feat_conv, ...
                    'label', gpuArray(single(labels)), ...
                    'label_score', gpuArray(single(labels)), ...
                    'action_history', gpuArray(action_history_oh_reshape)};
                net_fc.eval(inputs);
                net_pred = squeeze(gather(net_fc.getVar('prediction').value));
                net_pred_score = squeeze(gather(net_fc.getVar('prediction_score').value));
                curr_score = net_pred_score(2);

                % choose highest rated action out of all of them
                [~, net_max_action] = max(net_pred(1:end));

                % check for tracking failure: curr_score did too badly
                if (curr_score < opts.failedThre)
    %             if (prev_score < -0.5 && curr_score < prev_score) || curr_score < -5.0
                    % tracking failed
                    is_negative = true;
                    curr_score = prev_score;
                    action_history(2:end) = action_history(1:end-1);
                    action_history(1) = 12;
                    cont_negatives = cont_negatives + 1;
                    break;
                end
                % get next bbox after action chosen
                curr_bbox{i} = do_action(curr_bbox{i}, opts, net_max_action, imSize);

                % when come back again (oscillating) check for stop action
                [~, ism] = ismember(round(bb_step), round(curr_bbox{i}), 'rows');
                if sum(ism) > 0 ...
                        && net_max_action ~= opts.stop_action
                    net_max_action = opts.stop_action;
                end

                % update action history for next step
                action_history(2:end) = action_history(1:end-1);
                action_history(1) = net_max_action;
                this_actions(move_counter) = 1;
                target_score = curr_score;

                if net_max_action == opts.stop_action                
                    break;
                end

                move_counter = move_counter + 1;
                prev_score = curr_score;
            end
        end

        % ------------------------------------
        % TRACKING FAILURE - REDETECTION
        % ------------------------------------
        if frameIdx > 1 && is_negative == true %&& (mod(cont_negatives, 3) == 0 || cont_negatives < 3)
            total_pos_data{i}{frameIdx} = single([]);
            total_neg_data{i}{frameIdx} = single([]);
            total_pos_action_labels{i}{frameIdx} = single([]);
            total_pos_examples{i}{frameIdx} = single([]);
            total_neg_examples{i}{frameIdx} = single([]);

            % re-detection
            % I think this is taking random boxes to see if any get close
            % to tracking again
            samples_redet = gen_samples('gaussian', curr_bbox_old, opts.redet_samples, opts,  min(1.5, 0.6 * 1.15^cont_negatives), opts.redet_scale_factor);
            net_conv.mode = 'test';
            feat_conv = get_conv_feature(net_conv, curr_img, samples_redet, opts);
            feat_conv = gpuArray(feat_conv);
            % evaluate the candidates
            net_fc.mode = 'test';
            num_feats = size(feat_conv, 4); labels = zeros(num_feats, 1);
    %         action_history_oh_old = action_history_oh_zeros;
            action_history_oh_old_reshape = reshape(action_history_oh_old, [1,1,numel(action_history_oh_old),1]);
            action_history_oh_old_reshape = repmat(action_history_oh_old_reshape , [1,1,1,num_feats]);
            inputs = {'x10', feat_conv, ...
                'label', gpuArray(single(labels)), ...
                'label_score', gpuArray(single(labels)), ...
                'action_history', gpuArray(action_history_oh_old_reshape)};
            net_fc.eval(inputs);
            red_score_pred = squeeze(gather(net_fc.getVar('prediction_score').value));
            [scores,idx] = sort(red_score_pred(2,:),'descend');
            target_score = mean(scores(1:5));
            if target_score > curr_score
                curr_bbox{i} = mean(samples_redet(idx(1:5),:));
            end

        end


        % ------------------------------------
        % TRACKING SUCCESS - GENERATE POS/NEG SAMPLES
        % ------------------------------------
        if frameIdx > 1 && (is_negative == false || target_score > opts.successThre)
            % pos and neg examples
            cont_negatives = 0;
            pos_examples = gen_samples('gaussian', curr_bbox{i}, opts.nPos_online*2, opts, opts.finetune_trans, opts.finetune_scale_factor);
            r = overlap_ratio(pos_examples,curr_bbox{i});
            pos_examples = pos_examples(r>opts.posThre_online,:);
            pos_examples = pos_examples(randsample(end,min(opts.nPos_online,end)),:);

            neg_examples = gen_samples('uniform', curr_bbox{i}, opts.nNeg_online*2, opts, 2, 5);
            r = overlap_ratio(neg_examples,curr_bbox{i});
            neg_examples = neg_examples(r<opts.negThre_online,:);
            neg_examples = neg_examples(randsample(end,min(opts.nNeg_online,end)),:);

            examples = [pos_examples; neg_examples];
            pos_idx = 1:size(pos_examples,1);
            neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

            % store total pos and neg data and action labels and example
            net_conv.mode = 'test';
            feat_conv = get_conv_feature(net_conv, curr_img, examples, opts);
            total_pos_data{i}{frameIdx} = feat_conv(:,:,:,pos_idx);
            total_neg_data{i}{frameIdx} = feat_conv(:,:,:,neg_idx);

            pos_action_labels = gen_action_labels(opts.num_actions, opts, pos_examples, curr_bbox{i});
            total_pos_action_labels{i}{frameIdx} = pos_action_labels;
            total_pos_examples{i}{frameIdx} = pos_examples';
            total_neg_examples{i}{frameIdx} = neg_examples';

            frame_window{i} = [frame_window{i}, frameIdx];

            if(numel(frame_window{i})>opts.nFrames_long)
                total_pos_data{i}{frame_window{i}(end-opts.nFrames_long)} = single([]);
                total_pos_action_labels{i}{frame_window{i}(end-opts.nFrames_long)} = single([]);
                total_pos_examples{i}{frame_window{i}(end-opts.nFrames_long)} = single([]);
            end
            if(numel(frame_window{i})>opts.nFrames_short)
                total_neg_data{i}{frame_window{i}(end-opts.nFrames_short)} = single([]);
                total_neg_examples{i}{frame_window{i}(end-opts.nFrames_short)} = single([]);
            end
        end


        % ---------------------
        % DO ONLINE FINE TUNING
        % ---------------------
        if mod(frameIdx, opts.finetune_interval) == 0  || is_negative == true %(is_negative == true && mod(cont_negatives, 3) == 1)
            
            % modified from mdnet_run.m
            if mod(frameIdx, opts.finetune_interval) == 0
                % long term update
                f_st = max(1,numel(frame_window{i})-opts.nFrames_long+1);
                pos_data = cell2mat(total_pos_data{i}(frame_window{i}(f_st:end)));
                pos_action_labels = cell2mat(total_pos_action_labels{i}(frame_window{i}(f_st:end)));
            else
                % short term update
                f_st = max(1,numel(frame_window{i})-opts.nFrames_short+1);
                pos_data = cell2mat(total_pos_data{i}(frame_window{i}(f_st:end)));
                pos_action_labels = cell2mat(total_pos_action_labels{i}(frame_window{i}(f_st:end)));
            end

            f_st = max(1,numel(frame_window{i})-opts.nFrames_short+1);
            neg_data = cell2mat(total_neg_data{i}(frame_window{i}(f_st:end)));

            opts.maxiter = opts.finetune_iters_online;
            opts.learningRate = 0.0001;

            % using net_fc_noah for fine tuning?
            net_fc_noah = copy(net_fc);
            net_fc_noah.removeLayer('concat');
            net_fc_noah.layers(7).inputs = 'x16';
            net_fc_noah.layers(8).inputs = 'x16';
            net_fc_noah.params(5).value(:,:,512+1:end,:) = [];
            net_fc_noah.params(7).value(:,:,512+1:end,:) = [];
            net_fc_noah.rebuild();
            net_fc_noah.move('gpu');
            net_fc.move('gpu');        

            [net_fc_noah, ~] =  train_fc_finetune_hem(net_fc_noah, opts, ...
                pos_data, neg_data, pos_action_labels);
            for fci = 1 : 8
                if fci == 5 || fci == 7
                    net_fc.params(fci).value(:,:,1:512,:) = net_fc_noah.params(fci).value;
                else
                    net_fc.params(fci).value = net_fc_noah.params(fci).value;
                end
            end

        end
        % print status
        curr_t = toc(frame_tic);
        if opts.printscreen == 1
            fprintf('[%04d/%04d] time: %f\n', frameIdx, vid_info.nframes, curr_t);
        end
        t_sum = t_sum + curr_t;
    
        % bbox results
        bboxes(i,frameIdx,:) = [frameIdx,idnums(i),curr_bbox{i}];
    end
    
    % ---------------------
    % Visualize
    % ---------------------
    if opts.visualize == true
        clf(res_fig);
        imshow(curr_img);
        for i = 1:length(idnums)
            rectangle('Position', vid_info.gt{i}(frameIdx,3:6), 'EdgeColor', [0,1,0], 'LineWidth', 2.0); % ground truth box
            text(vid_info.gt{i}(frameIdx,3), vid_info.gt{i}(frameIdx,4), num2str(vid_info.gt{i}(frameIdx,2)), 'Color', 'y', 'FontSize', 12);
            rectangle('Position', curr_bbox{i},'EdgeColor', [0,0,1],'LineWidth',2.0); % generated box
            text(double(curr_bbox{i}(1)), double(curr_bbox{i}(2)), num2str(vid_info.gt{i}(frameIdx,2)), 'Color', 'y', 'FontSize', 12);
        end
        set(gca,'position',[0 0 1 1]);
        text(20,10,num2str(vid_info.img_files(frameIdx).name),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
        hold off;
        drawnow;
        saveas(gcf, fullfile(vid_path, 'img_bbox_folders', folder_name, sprintf('%04d.jpg',frameIdx)));
    end
    
end

% PRECISION RESULTS
positions = bboxes(:,:, [4,3]) + bboxes(:,:, [6,5]) / 2;
ground_truth=cell(length(idnums));
precisions=cell(length(idnums));
for i = 1:length(idnums)
    ground_truth{i} = vid_info.gt{i}(:,[4,3]) + vid_info.gt{i}(:,[6,5]) / 2;
    precisions{i} = precision_plot(positions(i), ground_truth{i}, '', 0);
end

% SAVE TO TXT
% TODO modify
dlmwrite(fullfile(vid_path, 'bbox_texts', sprintf('%s_bbox.txt', folder_name)), bboxes);
dlmwrite(fullfile(vid_path, 'precisions', sprintf('%s_prec.txt', folder_name)), precisions);

jpgToAvi(vid_path, fullfile('img_bbox_folders', folder_name), 'avi_vids', folder_name, 15/interval);
