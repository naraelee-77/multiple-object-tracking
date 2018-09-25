function [pre_vid_info] =  get_pre_vid_infos_lair(vid_path, vid_name, colors)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

pre_vid_info.img_files = [];
pre_vid_info.name = vid_name;
if(~exist(vid_path,'dir'))
    error('%s does not exist!!',vid_path);
end
img_files = dir(fullfile(vid_path,'*.pgm'));
for i = 1 : numel(img_files)
    pre_vid_info.img_files(i).name = fullfile(vid_path, img_files(i).name);
end

pre_vid_info.nframes = length(pre_vid_info.img_files);

switch vid_name
    case {'AOI01', 'AOI02', 'AOI03', 'AOI04'}
        % gt path
        gt_path = fullfile(get_parent_folder(vid_path), 'finalGT');
        if(~exist(gt_path,'file'))
            error('%s does not exist!!',gt_path);
        end
        
        [ids,bboxes]=LAIR_get_IDs_bboxes(gt_path);                     
    case {'AOI34', 'AOI40', 'AOI41', 'AOI42'}
        load(fullfile(get_parent_folder(vid_path), sprintf('GT_%s.mat', vid_name(4:5))));
        ids=cat(2,Data{:,1});
        bboxes=[];
        for i = 1:size(Data,1)
            bboxes(i,:,:)=Data{i,2};
        end
        bboxes(bboxes==-1)=0;
        
        bboxes(bboxes~=0)=bboxes(bboxes~=0)-50;
        bboxes(:,:,3:4)=100;
           
end

pre_vid_info.ids=ids;
pre_vid_info.bboxes=bboxes;
pre_vid_info.nids=numel(ids);

% res_fig = figure(1);
% set(res_fig,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
% 
% new_folder=
% 
% if ~exist(
% 
% for frame = 1:42
%     curr_img=imread(pre_video_info.img_files(frame).name);
%     clf(res_fig);
%     imshow(curr_img);
%     for i = 1:pre_video_info.nids
%         rectangle('Position', bboxes(i,frame,:), 'EdgeColor', colors(i,:), 'LineWidth', 0.25); % ground truth box
%     end
%     set(gca,'position',[0 0 1 1]);
%     saveas(gcf, fullfile(get_parent_folder(video_path), 'test.jpg'));
%     hold off;
% end

end
