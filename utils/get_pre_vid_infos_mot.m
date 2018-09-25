function [pre_vid_info] =  get_pre_vid_infos_mot(vid_path, vid_name, colors)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

pre_vid_info.img_files = [];
pre_vid_info.name = vid_name;
if(~exist(vid_path,'dir'))
    error('%s does not exist!!',vid_path);
end
img_files = dir(fullfile(vid_path, vid_name, 'img1', '*.jpg'));
for i = 1 : numel(img_files)
    pre_vid_info.img_files(i).name = fullfile(vid_path, vid_name, 'img1', img_files(i).name);
end

pre_vid_info.nframes = length(pre_vid_info.img_files);

gt_path = fullfile(vid_path, vid_name, 'gt', 'gt.txt');   % TODO
gt=load(gt_path);
gt=gt(:,1:6);

ids=[];
for line=1:size(gt,1)
    curr_id=gt(line,2);
    curr_frame=gt(line,1);
    
    if ~any(curr_id==ids)
        ids=[ids,curr_id];
    end
    bboxes(curr_id==ids, curr_frame,:)=gt(line, 3:6);
end

for curr_id=ids
    frames(curr_id==ids,:)=bboxes(curr_id==ids,:,1) & bboxes(curr_id==ids,:,2);
end

pre_vid_info.ids=ids;
pre_vid_info.bboxes=bboxes;
pre_vid_info.frames=frames;
pre_vid_info.nids=numel(ids);

end
