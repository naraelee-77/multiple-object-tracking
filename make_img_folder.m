function folder_name=make_img_folder(vid_path, idnum, begin_frame, end_frame, interval)

folder_name=sprintf('img_id%01d_n%01d', idnum, interval);

if ~exist(fullfile(vid_path, folder_name), 'dir')
    fprintf('Making folder %s...\n', fullfile(vid_path, folder_name));
    mkdir(vid_path, folder_name);
    
    for i = begin_frame:interval:end_frame
        copyfile(fullfile(vid_path, 'img', sprintf('%04d.jpg', i)), fullfile(vid_path, folder_name, sprintf('%04d.jpg', i)));
    end
else
    fprintf('Folder %s already exists\n', fullfile(vid_path, folder_name));
end
end
