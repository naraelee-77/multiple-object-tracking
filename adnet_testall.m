function adnet_testall(model)

%     models = {'net_13_14_15', 'net_16_17'}; %, 'net_orig'}; % 'net_13_14_15', 
    num_ids=19;
    intervals = {1, 2, 5, 8};
    
%     for m = 1:numel(models)
        for id = 2:num_ids
            for n = 1:numel(intervals)
                fprintf('Testing ID%01d at interval %01d\n', id, intervals{n});
                adnet_demo(model, 'PETS', id, intervals{n});
            end
        end
%     end
            
%     data_samples = importdata(fullfile('data', 'otb-list2.txt'));
% 
%     for j = 1 : numel(data_samples)
%         for i = 1 : numel(models)
%             fprintf('Testing %s model with %s data sample:\n', models{i}, fullfile('data', data_samples{j}));
%             adnet_demo(models{i}, fullfile('data', data_samples{j}));
%         end
%     end
end