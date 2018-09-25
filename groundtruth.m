function [begin_frame, end_frame] = groundtruth(vid_path, idnum, interval)

% if nargin < 2
filename = fullfile(vid_path, 'gt.txt');
% end
if nargin < 3
    interval = 1;
end

gt = load(filename);
target = sprintf('gt_id%01d_n%01d.txt',idnum,interval);
fprintf('Creating ground truth file %s...\n', target);
% if ~exist(fullfile(vid_path, target),'file')
    gt_id = fopen(fullfile(vid_path, target),'w');
% end

begin_frame=0;
end_frame=0;

for i = 1:length(gt)
    if gt(i,2) == idnum
        curr_frame=gt(i,1);
        if begin_frame==0
            begin_frame=gt(i,1);
        end
        if mod(curr_frame - begin_frame, interval) == 0
            fprintf(gt_id, '%g,%g,%g,%g\n', gt(i,3), gt(i,4), gt(i,5), gt(i,6));
        end
        end_frame=gt(i,1);
    end
end
        
fclose(gt_id);
end