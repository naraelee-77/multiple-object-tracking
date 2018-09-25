function vid_info =  get_vid_infos_mot_v2(pre_vid_info, ids)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

threshold=32;
which_frames=ones(pre_vid_info.nframes,1);
vid_info=[];

for curr_id = ids
    which_frames=which_frames & pre_vid_info.frames(:, curr_id==pre_vid_info.ids);
end

if sum(which_frames) < threshold
    return;
end

vid_info.name=pre_vid_info.name;
vid_info.img_files=pre_vid_info.img_files(which_frames);

bboxes=zeros(sum(which_frames),4,numel(ids));
for i = 1:numel(ids)
    bboxes(:,:,i)=pre_vid_info.bboxes(which_frames, :, ids(i)==pre_vid_info.ids);
end

vid_info.gt=bboxes; % nframes x 4 x nids
vid_info.nframes=size(bboxes,1);
vid_info.ids=ids;
vid_info.nids=numel(ids);
end
