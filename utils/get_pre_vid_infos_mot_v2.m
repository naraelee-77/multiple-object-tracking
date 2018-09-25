function [pre_vid_info] =  get_pre_vid_infos_mot_v2(vid_path, vid_name)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

img_path = fullfile(vid_path, vid_name, 'img1');

if(~exist(img_path,'dir'))
    error('%s does not exist!!',vid_path);
end

pre_vid_info.img_files = [];
pre_vid_info.name = vid_name;

img_files = dir(fullfile(img_path, '*.jpg'));
for i = 1 : numel(img_files)
    pre_vid_info.img_files(i).name = fullfile(img_path, img_files(i).name);
end
pre_vid_info.nframes = length(pre_vid_info.img_files);

gt_path = fullfile(vid_path, vid_name, 'gt', 'gt.txt');
gt=load(gt_path);
gt=gt(:,1:6);

ids=[];
for line=1:size(gt,1)
    curr_id=gt(line,2);
    curr_frame=gt(line,1);
    
    if ~any(curr_id==ids)
        ids=[ids,curr_id];
    end
    bboxes(curr_frame, :, curr_id==ids)=gt(line, 3:6);
end

for curr_id=ids
    frames(:, curr_id==ids)=bboxes(:, 1, curr_id==ids) & bboxes(:, 2, curr_id==ids);
end

pre_vid_info.ids=ids;   % 1 x nids
pre_vid_info.nids=numel(ids);
pre_vid_info.bboxes=bboxes; % nframes x 4 x nids
pre_vid_info.frames=frames; % nframes x nids

end
