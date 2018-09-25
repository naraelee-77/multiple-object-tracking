function jpgToAvi(img_path, vid_name, frame_rate)
    
fprintf('Converting to avi');
v=VideoWriter(fullfile(get_parent_folder(img_path), vid_name));
v.FrameRate=frame_rate;
open(v)

% disp(ls(fullfile(vid_path, folder_name)));
image_names = dir(fullfile(img_path, '*.jpg'));
image_names = {image_names.name};
% disp(image_names);

for i = 1:length(image_names)
    img = imread(fullfile(img_path, image_names{i}));
    writeVideo(v,img);
    fprintf('.');
end
fprintf('\n');

close(v)

end