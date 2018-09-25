function [vid_info] =  get_vid_infos_lair(pre_vid_info, i)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

vid_info.name=pre_vid_info.name;
which_frames=pre_vid_info.bboxes(i,:,1) & pre_vid_info.bboxes(i,:,2);
vid_info.img_files=pre_vid_info.img_files(which_frames);
bboxes=reshape(pre_vid_info.bboxes(i,:,:),size(pre_vid_info.bboxes,2),size(pre_vid_info.bboxes,3));
bboxes=bboxes(which_frames,:);
vid_info.gt=bboxes;
vid_info.nframes=size(bboxes,1);
end
