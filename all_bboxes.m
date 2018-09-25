function all_bboxes(vid_path, model_name, n)

if nargin < 3
    n = 1;
end

if ~exist(fullfile(vid_path, 'img_boxes2'), 'dir')
    mkdir(fullfile(vid_path, 'img_boxes2'));
end

img_files = dir(fullfile(vid_path, 'img', '*.jpg'));
gt = load(fullfile(vid_path, 'gt.txt'));
bb_file = fopen(fullfile(vid_path, 'bounding_boxes.txt'), 'w');

bboxes = zeros(19, numel(img_files), 4);    % find out how to initialize this
id_begin_end = zeros(19, 2);

for line = 1:size(gt,1)
    idnum = gt(line, 2);
    if id_begin_end(idnum, 1) == 0
        id_begin_end(idnum, 1) = gt(line, 1);
    end
    id_begin_end(idnum, 2) = gt(line, 1);
end

for id = 1:19
    bb = load(fullfile(vid_path, sprintf('%s_id%01d_n%01d_bbox.txt', model_name, id, n)));
    bboxes(id,:,:) = cat(1, zeros(id_begin_end(id, 1)-1,4), bb, zeros(numel(img_files)-id_begin_end(id, 2),4));
end

count = 0;

for frame = 1:numel(img_files)
    for id = 1:19
        if any(bboxes(id, frame, :))
            fprintf(bb_file, '%01d,%01d,%g,%g,%g,%g\n', frame, id, bboxes(id, frame, 1), bboxes(id, frame, 2), bboxes(id, frame, 3), bboxes(id, frame, 4));
            count = count + 1;
        end
    end
end

fprintf('Number of lines: %d\n', count);

fclose(bb_file);

bb_all = load(fullfile(vid_path, 'bounding_boxes.txt'));

line = 1;

for frame = 1:numel(img_files)
    res_fig = figure(1);
    set(res_fig,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    clf(res_fig);
    
    curr_img = imread(fullfile(vid_path, 'img', img_files(frame).name));
    imshow(curr_img);
    while gt(line, 1) == frame && bb_all(line,1) == frame
        rectangle('Position', gt(line,3:6), 'EdgeColor', [0,1,0], 'LineWidth', 2.0); % ground truth box
        text(gt(line,3), gt(line,4), num2str(gt(line,2)), 'Color', 'y', 'FontSize', 12);
        rectangle('Position', bb_all(line,3:6),'EdgeColor', [0,0,1],'LineWidth',2.0); % generated box
        text(bb_all(line,3), bb_all(line,4), num2str(bb_all(line,2)), 'Color', 'y', 'FontSize', 12);
        line = line + 1;
    end
    
    set(gca,'position',[0 0 1 1]);
    hold off;
    drawnow;
    saveas(gcf, fullfile(vid_path, 'img_boxes2', sprintf('%04d.jpg', frame)));
end

end