function [curr_bboxes, action_history, this_actions, ims, actions, ah_onehot, fr_toc] = ...
    tracker_rl_mot(curr_img, curr_bboxes, my_net, action_history, opts)
% TRACKER_RL
% 
% Sangdoo Yun, 2017.


fr_tic = tic;

move_counter = 1;   % alg 2 line 1
num_action_step_max = opts.num_action_step_max;
num_show_actions = opts.num_show_actions;
imSize = opts.imgSize;
bb_step = zeros(3, num_action_step_max, 4);
this_actions = zeros(num_show_actions,3);
% conv_feats =  zeros(3,3,512, num_action_step_max, 'single');
curr_score = zeros(1,3);
prev_score = zeros(1,3)-inf;

ims = zeros([112,112,3,num_action_step_max], 'single');
ah_onehot = zeros(opts.num_actions*opts.num_action_history, num_action_step_max, 3);
actions = zeros(3, num_action_step_max);

while (move_counter <= num_action_step_max)
    bb_step(:, move_counter,:) = curr_bboxes;
    
    my_net.mode = 'test';
    im1 = get_extract_regions(curr_img, curr_bboxes(1,:), opts);
    im2 = get_extract_regions(curr_img, curr_bboxes(2,:), opts);
    im3 = get_extract_regions(curr_img, curr_bboxes(3,:), opts);
    ims(:,:,:,:,move_counter) = cat(4, im1, im2, im3);

    ah_ohs = zeros(numel(ah_oh), 3, 'logical');
    for id = 1:3
        ah_oh = get_action_history_onehot(action_history(1:opts.num_action_history, id), opts); % get 10 most recent actions in onehot form
        ah_oh = reshape(ah_oh, [1,1,numel(ah_oh),1]);
        ah_ohs(id, :) = ah_oh;
        ah_onehot(:,move_counter, id) = ah_oh;
    end
    
    inputs = {'input_id1', im1, 'input_id2', im2, 'input_id3', im3, ...
        'action_history', gpuArray(ah_ohs)};
    my_net.eval(inputs);
    
    tostop = true;
    for id = 1:3
        my_net_pred = squeeze(gather(my_net.getVar(sprintf('prediction_id%d', id)).value));
        my_net_pred_score = squeeze(gather(my_net.getVar(sprintf('prediction_score_id%d', id)).value));
        curr_score(id) = my_net_pred_score(2);
        [~, my_net_max_action] = max(my_net_pred(1:end));   % alg 2 line 7
        if (prev_score(id) < -5.0 && curr_score(id) < prev_score(id))
            if randn(1) < 0.5
                my_net_max_action = randi(11);
            end
        end
        actions(id,move_counter) = my_net_max_action;
        curr_bboxes(id,:) = do_action(curr_bboxes(id,:), opts, my_net_max_action, imSize);  % alg 2 line 8

        % when come back again (oscillating)
        [~, ism] = ismember(round(bb_step(id, :,:)), round(curr_bboxes(id,:)), 'rows');
        if sum(ism) > 0 ...
                && my_net_max_action ~= opts.stop_action        
            my_net_max_action = opts.stop_action;   % TODO: stop one action at a time
        else
            tostop = false;
        end    
        
        % alg 2 line 10, update ah
        action_history(2:end, id) = action_history(1:end-1, id);
        action_history(1, id) = my_net_max_action;
        this_actions(move_counter, id) = 1;
    end    
    
    % alg 2 line 13
    if tostop
        break;
    end    
    move_counter = move_counter + 1;    % alg 2 line 12
    prev_score = curr_score;
end
% alg 2 line 14
actions(:,move_counter+1:end) = [];
% conv_feats(:,:,:,move_counter+1:end) = [];
ims(:,:,:,:, move_counter+1:end) = [];
ah_onehot(:, move_counter+1:end) = [];
fr_toc = toc(fr_tic);
