function [bboxes, t_sum, precisions, successes] = adnet_test_mult_LAIR(title, model_name, color, path, idnums)
% ADNET_DEMO Run tracking on a sequence
%
% Sangdoo Yun, 2017.


path='../LAIR_CLIF_PV/LAIR/AOI02';
img_folder='finalCrop';
gt_folder='finalGT';
% model_name='net_13_14_15';
idnums = [88,89,197,199,195,193,196,312,345,320,316,349,329,311,534,535,529,513,520,515,540,548,516,522,530,526,517,512,538,543,582,604,602,616,586,609,618,607,612,576,619,597,556,603,577,600,574,583,601,575,579,625,622,627,658,710,704,706,711,2164,1927,2035,2121,2158,2170,2360,2364,2406,2333,2247,2298,2256,2502,3077,2896,3120,6650,4915,5706,6221,6567,6551,198,8197,7341,7505,7515,7201,8688,7855,7861,8004,8113,9253,9075,9167,9294,9664,11342,10471,36671,21134,27697,25656,28154];
% idnums=[2,11,15,16,17,18,19,21,22,20,23,24,54,66,61,59,65,55,63,214,215,217,213,309,422,402,391,436,433,366,310,306,384,308,357,307,555,446,560,541,537,558,445,498,527,504,531,509,584,709,635,650,695,680,722,689,671,653,668,692,693,713,685,674,681,718,719,715,721,716,732,725,747,723,731,797,792,789,796,794,804,769,790,808,800,805,795,803,793,811,821,817,815,2183,813,836,834,820,816,825,832,914,833,959,956,963,1238,1242,1245,1247,1252,1255,1260,1265,1269,11256,801,1821,1844,443,1865,1853,1867,1869,426,2134,2136,2137,2227,2124,2131,2027,2023,2040,2132,2128,726,1942,6060,2112,2412,2231,2534,2571,2532,2531,2590,6660,2522,2429,2441,27159,2494,2510,2512,437,2546,2883,3089,2729,2855,3133,3130,3166,3234,4676,4901,4560,4924,4934,5215,5218,5038,5041,5062,6063,5736,5740,5742,6258,62,638,6467,6190,5845,1843,6021,6261,6530,6401,7067,6580,6634,6746,6635,6846,6691,6692,6656,6633,6604,6608,7032,7187,7100,6758,6744,6814,6808,7349,8103,8105,8009,7900,7950,8198,8926,8928,8931,8583,9055,9612,9613,9002,9271,9054,9288,8980,9261,9648,29162,9649,9615,9611,9614,11047,809,10341,12086,58,12095,10839,11117,12807,12851,12388,14028,14930,14131,14934,15626,15377,32248,18050,18505,18417,18943,18971,2420,18547,26763,4621,829,27660,31678,35964,32868,32575,56,57,15875,36326,761,15876,2413,2418,2565];
if nargin < 3
    color=rand(numel(idnums),3);
end

if ~exist(fullfile(path,title),'dir')
    mkdir(fullfile(path,title));
end
if ~exist(fullfile(path,title,'img'),'dir')
    mkdir(fullfile(path,title,'img'));
end

addpath('test/');
addpath(genpath('utils/'));

init_settings;

run(matconvnet_path);

load(fullfile('models', strcat(model_name,'.mat')));

opts.visualize = true;
opts.printscreen = true;

% if ~exist(fullfile(path, 'gt'), 'dir')
%     mkdir(fullfile(path, 'gt'));
% end

fprintf('Testing sequences in %s... \n', fullfile(path,img_folder));
t_sum = 0;

% ============================
% Load video sequences
% ============================
vid_info.img_files = dir(fullfile(path, img_folder, '*.pgm'));
vid_info.gt_files = dir(fullfile(path, gt_folder, '*.csv'));
% 
% disp(vid_info.img_files);
% disp(vid_info.gt_files);

vid_info.nframes = numel(vid_info.img_files);
% vid_info.gt=cell(numel(idnums));
% 
% for i = 1:numel(idnums)
%     vid_info.gt{i}=zeros(vid_info.nframes,5);
% end
vid_info.gt=zeros(numel(idnums),vid_info.nframes,5);

for i = 1:vid_info.nframes
    gt=load(fullfile(vid_info.gt_files(i).folder,vid_info.gt_files(i).name));
    for j = 1:length(gt)
        id = gt(j,1);
        if ismember(id, idnums)
            vid_info.gt(id==idnums,i,1:3)=gt(j,:);
        end
    end    
end

% disp(vid_info.gt{1});
% disp(size(vid_info.gt));
% for i = 1:numel(idnums)
% %     vid_info.gt{i}(:,2:3)=vid_info.gt{i}(:,2:3)-5;
% %     vid_info.gt{i}(:,4:5)=zeros(size(vid_info.gt{i}(:,2:3)));
% %     vid_info.gt{i}(:,4:5)=vid_info.gt{i}(:,4:5)+10;
%     vid_info.gt{i}(:,2:3)=vid_info.gt{i}(:,2:3)-12;
% %     vid_info.gt{i}(:,3)=vid_info.gt{i}(:,3)-12;
%     vid_info.gt{i}(:,4:5)=25;
% %     vid_info.gt{i}(:,5)=25;
% end

vid_info.gt(:,:,2:3)=vid_info.gt(:,:,2:3)-12;
vid_info.gt(:,:,4:5)=25;

% folder_name=sprintf('%s_%s_ids%s', 'LAIR', model_name, sprintf('_%01d', idnums(:)));
% if ~exist(fullfile(vid_path, 'img_bbox_folders', folder_name), 'dir')
%     mkdir(fullfile(vid_path, 'img_bbox_folders', folder_name));
% end

% init containers
bboxes = zeros(size(vid_info.gt));
total_pos_data = cell(length(idnums));
total_neg_data = cell(length(idnums));
total_pos_action_labels = cell(length(idnums));
total_pos_examples = cell(length(idnums));
total_neg_examples = cell(length(idnums));
% 
for i = 1:length(idnums)
    total_pos_data{i} = cell(1,1,1,vid_info.nframes);
    total_neg_data{i} = cell(1,1,1,vid_info.nframes);
    total_pos_action_labels{i} = cell(1,vid_info.nframes);
    total_pos_examples{i} = cell(1,vid_info.nframes);
    total_neg_examples{i} = cell(1,vid_info.nframes);
%     color(i,:)=[rand(),rand(),rand()];
end

% init model networks
net.layers(net.getLayerIndex('loss')).block.loss = 'softmaxlog';
[net, net_conv, net_fc] = split_dagNN(net);

net_fc.params(net_fc.getParamIndex('fc6_1b')).value = ...
    gpuArray(ones(size(net_fc.params(net_fc.getParamIndex('fc6_1b')).value), 'single') * 0.01);

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
for frameIdx = 1 : 30 % vid_info.nframes
    
    for i = 1:length(idnums)
    
        if frameIdx == 1

            % ============================
            % LOAD & FINETUNE FC NETWORKS
            % ============================
            curr_bbox{i} = reshape(vid_info.gt(i,frameIdx,2:5),1,4);
            curr_img = imread(fullfile(vid_info.img_files(1).folder,vid_info.img_files(1).name));
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
            curr_img = imread(fullfile(vid_info.img_files(frameIdx).folder, vid_info.img_files(frameIdx).name));
            fprintf('%s\n',fullfile(vid_info.img_files(frameIdx).folder, vid_info.img_files(frameIdx).name));
            if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end
            curr_bbox_old = curr_bbox{i};
            frame_tic = tic;
            move_counter = 1;
            target_score = 0;

            num_action_step_max = 20;
            bb_step = zeros(num_action_step_max, 4);
            score_step = zeros(num_action_step_max, 1);
            is_negative = false;
            prev_score = -inf;
            this_actions = zeros(opts.num_show_actions,1);
            action_history_oh_old = action_history_oh;
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
                inputs = {'x10', curr_feat_conv, ...
                    'label', gpuArray(single(labels)), ...
                    'label_score', gpuArray(single(labels)), ...
                    'action_history', gpuArray(action_history_oh_reshape)};
                net_fc.eval(inputs);
                net_pred = squeeze(gather(net_fc.getVar('prediction').value));
                net_pred_score = squeeze(gather(net_fc.getVar('prediction_score').value));
                curr_score = net_pred_score(2);

                [~, net_max_action] = max(net_pred(1:end));

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
                curr_bbox{i} = do_action(curr_bbox{i}, opts, net_max_action, imSize);

                % when come back again (oscillating)
                [~, ism] = ismember(round(bb_step), round(curr_bbox{i}), 'rows');
                if sum(ism) > 0 ...
                        && net_max_action ~= opts.stop_action
                    net_max_action = opts.stop_action;
                end


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
        curr_t = toc(frame_tic);
        if opts.printscreen == 1
            fprintf('[%04d/%04d] time: %f\n', frameIdx, vid_info.nframes, curr_t);
        end
        t_sum = t_sum + curr_t;
    
        % bbox results
        bboxes(i,frameIdx,:) = [idnums(i),curr_bbox{i}];
    end
    
    % ---------------------
    % Visualize
    % ---------------------
%     curr_img = imread(fullfile(vid_info.img_files(1).folder,vid_info.img_files(1).name));
    if opts.visualize == true
%         fig=figure;
        clf(res_fig);
        imshow(curr_img);
%         truesize
        for i = 1:length(idnums)
            
%             rectangle('Position', vid_info.gt{i}(frameIdx,2:5), 'EdgeColor', color(i,:), 'LineWidth', 1.0); % ground truth box
%             text(vid_info.gt{i}(frameIdx,2), vid_info.gt{i}(frameIdx,3), num2str(vid_info.gt{i}(frameIdx,1)), 'Color', 'y', 'FontSize', 12);
            rectangle('Position', curr_bbox{i},'EdgeColor', color(i,:),'LineWidth',2.0); % generated box
%             text(double(curr_bbox{i}(1)), double(curr_bbox{i}(2)), num2str(vid_info.gt{i}(frameIdx,1)), 'Color', 'y', 'FontSize', 12);
        end
        set(gca,'position',[0 0 1 1]);
%         text(20,10,num2str(vid_info.img_files(frameIdx).name),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
%         drawnow;
        saveas(gcf, fullfile(path, title, 'img', sprintf('%04d.jpg', frameIdx)));
%         frm=getframe(curr_img);
%         imwrite(frm.cdata, fullfile(path, 'test.pgm'));
        hold off;
    end
    
end

% PRECISION RESULTS
% positions = bboxes(:,:, [3,2]) + bboxes(:,:, [5,4]) / 2;
% ground_truth = vid_info.gt(:,:, [3,2]) + vid_info.gt(:,:, [5,4]) / 2;
% precisions=precision_plot(positions, ground_truth, 'LAIR050', 1);
precisions=precision_plot(bboxes(:,:,2:5), vid_info.gt(:,:,2:5), title, fullfile(path, title, 'precisions.jpg'), 1);
successes=success_plot(bboxes(:,:,2:5), vid_info.gt(:,:,2:5), title, fullfile(path, title, 'successes.jpg'), 1);
% ground_truth=cell(length(idnums));
% precisions=cell(length(idnums));
% for i = 1:length(idnums)
%     ground_truth{i} = vid_info.gt(i,:,[3,2]) + vid_info.gt(i,:,[5,4]) / 2;
%     precisions{i} = precision_plot(positions(i,:,:), ground_truth{i}, '', 1);
% end

% SAVE TO TXT
% TODO modify
% if ~exist(fullfile(path, 'bbox_texts'), 'dir')
%     mkdir(fullfile(path, 'bbox_texts'));
% end
% if ~exist(fullfile(path, 'precisions'), 'dir')
%     mkdir(fullfile(path, 'precisions'));
% end

dlmwrite(fullfile(path, title, 'bboxes.txt'), []);
for i = 1:numel(idnums)
    dlmwrite(fullfile(path, title, 'bboxes.txt'), reshape(bboxes(i,:,:),size(bboxes,2), size(bboxes,3)), '-append');
end
dlmwrite(fullfile(path, title, 'precisions.txt'), precisions);
dlmwrite(fullfile(path, title, 'successes.txt'), successes);

jpgToAvi(fullfile(path, title, 'img'), title, 10);
