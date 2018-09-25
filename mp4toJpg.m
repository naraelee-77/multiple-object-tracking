function frame_rate = mp4toJpg(vid_path, folder_name, vid_name)

fprintf(fullfile(vid_path, strcat(vid_name, '.mp4')));
v=VideoReader(fullfile(vid_path, strcat(vid_name, '.mp4')));
frame_rate=v.FrameRate;

if ~exist(fullfile(vid_path, folder_name), 'dir')
    mkdir(fullfile(vid_path, folder_name));
    
    i=1;

    while hasFrame(v)
        curr_frame = readFrame(v);
        imwrite(curr_frame, fullfile(vid_path, folder_name, sprintf('%04d.jpg',i)));
        i=i+1;
    end
    
end
end