function [ids,bboxes] = LAIR_get_IDs_bboxes(gt_path)

gt_files=dir(fullfile(gt_path, '*.csv'));

if numel(gt_files)==0
    fprintf('No files found. Exiting.\n');
end

fr=1;
curr=load(fullfile(gt_files(fr).folder, gt_files(fr).name));
ids=[];
bboxes=[];

ids=curr(:,1).';    % ids = 1 x numids
bboxes(:,1,:)=curr(:,2:3);   % bboxes = numids x numframes x 2 (will be 6)
% fprintf('.');
    
for fr = 2:numel(gt_files)
    curr=load(fullfile(gt_files(fr).folder, gt_files(fr).name));
    for i = 1:size(curr, 1)
        id=curr(i,1);
        if ~any(id==ids)
            ids(numel(ids)+1)=id;
        end
        bboxes(id==ids,fr,:)=curr(i,2:3);
    end
%     fprintf('.');
    if mod(fr,80)==0
%         fprintf('\n');
    end
end
% fprintf('\n');

% for i = 1:size(bboxes,1)
%     for j = 1:size(bboxes,2)-1
%         if all(bboxes(i,j:j+1,1:2))
%             del_x = abs(bboxes(i,j+1,1)-bboxes(i,j,1));
%             del_y = abs(bboxes(i,j+1,2)-bboxes(i,j,2));
%             if del_x == 0 && del_y == 0
%                 bboxes(i,j,5:6)=[25,25];
%             else
%                 hyp = sqrt(del_x^2+del_y^2);
%                 height = (25*del_y+11*del_x)/hyp;
%                 width = (25*del_x+11*del_y)/hyp;
%                 bboxes(i,j,5:6) = [height,width];
%             end          
%         end
%     end
% end

% bboxes(:,:,3:4) = bboxes(:,:,1:2) - bboxes(:,:,5:6)/2;
% bboxes=round(bboxes);
bboxes(bboxes~=0)=bboxes(bboxes~=0)-12;
bboxes(:,:,3:4)=25;

% res_fig = figure(1);
% set(res_fig,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
% 
% curr_img=imread(fullfile(parent(gt_path),'finalCrop','cropImage10001.pgm'));
% clf(res_fig);
% imshow(curr_img);
% 
% for i = 1:numel(ids)
%     if all(bboxes(i,1,:))
%         rectangle('Position', [bboxes(i,1,1)-12,bboxes(i,1,2)-12,25,25], 'EdgeColor', [rand(), rand(), rand()], 'LineWidth', 0.25); % ground truth box
%     end
% end
% 
% set(gca,'position',[0 0 1 1]);
% drawnow;
% saveas(gcf, fullfile(parent(gt_path), 'test.jpg'));
% hold off;

end