function [train_db, all_vid_info] = make_train_db(train_db_path, vid_path, vid_names, opts)
if ~exist(train_db_path, 'file')
    fprintf('Num videos: %d\n', opts.num_videos);
    train_db={};
    all_vid_info={};
    for vid_idx = 1 : opts.num_videos
        vid_name = vid_names(vid_idx).name;
        pre_vid_info = get_pre_vid_infos_mot_v2(vid_path, vid_name);
        id_combos = combnk(pre_vid_info.ids, 3);
        combo_order=randperm(size(id_combos,1));
        id_combos=id_combos(combo_order,:);
        
        count=1;
        for num = 1:size(id_combos,1)
            rand_ids = id_combos(num,:);
            vid_info = get_vid_infos_mot_v2(pre_vid_info, rand_ids);
            if isempty(vid_info)
                continue;
            end

            train_db{end+1} = get_train_dbs_mot_v2(vid_info, opts);
            all_vid_info{end+1} = vid_info;
            fprintf('Video %d\tIDs [%s]\tcount %d\n', vid_idx, sprintf('%d ', rand_ids), count);
            
            if count >= 30
                break;
            end
            count=count+1;
        end
    end
    save(train_db_path, 'train_db', 'all_vid_info', '-v7.3');
else
    load(train_db_path);
end