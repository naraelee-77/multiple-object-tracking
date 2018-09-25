function [curr_bboxes, action_history, this_actions, conv_feats, ims, actions, ah_threehot, fr_toc] = ...
    tracker_rl_mot_v2(curr_img, curr_bboxes, my_net_conv, my_net_fc, action_history, opts)
% TRACKER_RL
% 
% Sangdoo Yun, 2017.


fr_tic = tic;

move_counter = 1;   % alg 2 line 1
num_action_step_max = opts.num_action_step_max;
num_show_actions = opts.num_show_actions;
imSize = opts.imgSize;
bb_step = zeros(num_action_step_max, 4, 3);
this_actions = zeros(num_show_actions,1);
conv_feats =  zeros(3,3,512,num_action_step_max, 'single');
prev_score = zeros(1,3)-inf;
curr_score = zeros(1,3);
my_net_max_action = zeros(1,3);

ims = zeros([112,112,6,num_action_step_max], 'single');

ah_threehot = zeros(3*opts.num_actions*opts.num_action_history, num_action_step_max);
actions = zeros(num_action_step_max, 3);

while (move_counter <= num_action_step_max)
    bb_step(move_counter,:,:) = curr_bboxes;  
    
    my_net_conv.mode = 'test';
    [curr_feat_conv, im] = get_conv_feature_v2(my_net_conv, curr_img, curr_bboxes, opts);
    ims(:,:,:, move_counter) = im;
    conv_feats(:,:,:, move_counter) = curr_feat_conv;
    curr_feat_conv = gpuArray(curr_feat_conv);
    
    my_net_fc.mode = 'test';
    labels = zeros(1);
    
    ah_th = [];
    for id = 1:3
        ah_oh = get_action_history_onehot(action_history(1:opts.num_action_history, id), opts); % get 10 most recent actions in onehot form
        ah_oh = reshape(ah_oh, [1,1,numel(ah_oh),1]);
        ah_th = cat(3, ah_th, ah_oh);   % th = threehot
    end
    ah_threehot(:,move_counter) = ah_th;
    inputs = {'x10', curr_feat_conv, 'action_history', gpuArray( ah_th)};
    my_net_fc.eval(inputs);
    
    for id = 1:3
        my_net_pred = squeeze(gather(my_net_fc.getVar(['prediction' num2str(id)]).value));
        my_net_pred_score = squeeze(gather(my_net_fc.getVar(['prediction_score' num2str(id)]).value));
        curr_score(id) = my_net_pred_score(2);
        [~, my_net_max_action(id)] = max(my_net_pred(1:end));   % alg 2 line 7
        if (prev_score(id) < -5.0 && curr_score(id) < prev_score(id))
            if randn(1) < 0.5
                my_net_max_action(id) = randi(11);
            end
        end
        actions(move_counter, id) = my_net_max_action(id);
        curr_bboxes(:,:,id) = do_action(curr_bboxes(:,:,id), opts, my_net_max_action(id), imSize);  % alg 2 line 8
        
        % when come back again (oscillating)
        [~, ism] = ismember(round(bb_step(:,:,id)), round(curr_bboxes(:,:,id)), 'rows');
        if (sum(ism) > 0 && my_net_max_action(id) ~= opts.stop_action) || action_history(1,id) == opts.stop_action      
            my_net_max_action(id) = opts.stop_action;
        end    
    end
    
    % alg 2 line 10, update ah
    action_history(2:end,:) = action_history(1:end-1,:);
    action_history(1,:) = my_net_max_action;
    this_actions(move_counter) = 1;
    
    % alg 2 line 13
    if all(my_net_max_action == opts.stop_action)
        break;
    end    
    move_counter = move_counter + 1;    % alg 2 line 12
    prev_score = curr_score;
end
% alg 2 line 14
actions(move_counter+1:end,:) = [];
conv_feats(:,:,:,move_counter+1:end) = [];
ims(:,:,:, move_counter+1:end) = [];
ah_threehot(:, move_counter+1:end) = [];
fr_toc = toc(fr_tic);
