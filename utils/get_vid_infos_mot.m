function [vid_info, success] =  get_vid_infos_mot(pre_vid_info, ids)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

threshold=32;
success = true;

vid_info.name=pre_vid_info.name;

which_frames=ones(1,pre_vid_info.nframes);
for curr_id = ids
    which_frames=which_frames & pre_vid_info.frames(curr_id==pre_vid_info.ids,:);
end

if sum(which_frames) < threshold
    success = false;
    return;
end

vid_info.img_files=pre_vid_info.img_files(which_frames);

bboxes=[];
for i = 1:numel(ids)
    bboxes(i,:,:)=pre_vid_info.bboxes(ids(i)==pre_vid_info.ids, which_frames, :);
end

vid_info.gt=bboxes;
vid_info.nframes=size(bboxes,2);
vid_info.nids=numel(ids);
end
