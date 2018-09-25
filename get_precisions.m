function get_precisions(model_name, n, vid_path)

if nargin < 3
    vid_path = 'data/PETS';
end

if ~isnumeric(n)
    n = str2num(n);
end

addpath('utils/');

file = fopen(fullfile(vid_path, sprintf('precisions_%s_n%01d.txt', model_name, n)), 'w');
sum = 0;

for id = 1:19
    gt = load(fullfile(vid_path, sprintf('gt_id%01d_n%01d.txt', id, n)));
    bb = load(fullfile(vid_path, sprintf('%s_id%01d_n%01d_bbox.txt', model_name, id, n)));
    prec = get_prec(gt, bb);
    fprintf(file, '%01d,%s\n', id, prec(end));
    sum = sum + prec(end);
end

avg_prec = sum/19;
fprintf(file, 'avg,%g\n', avg_prec);
fprintf('Average for model %s at n=%01d: %g\n', model_name, n, avg_prec);

fclose(file);

function precisions = get_prec(ground_truth, bboxes)

positions_bb = bboxes(:, [2,1]) + bboxes(:, [4,3]) / 2;
positions_gt = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
precisions = precision_plot(positions_bb, positions_gt, '', 0);

end

end