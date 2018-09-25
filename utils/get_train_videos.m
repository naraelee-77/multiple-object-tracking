function train_db = get_train_videos(opts)
% GET_TRAIN_VIDEOS 
% 
% Sangdoo Yun, 2017.

train_db_names = opts.train_dbs;

video_names = {};
video_paths = {};
for dbidx = 1 : numel(train_db_names)
    video_names{end+1}=train_db_names{dbidx};
    video_paths{end+1}=fullfile('../LAIR_CLIF_PV/LAIR', train_db_names{dbidx}, 'finalCrop');
end

train_db.video_names = video_names;
train_db.video_paths = video_paths;
end
