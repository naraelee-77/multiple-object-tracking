function [t,p, results] = adnet_demo (vid_name, model_name, idnum, interval)
% ADNET_DEMO Demonstrate `action-decision network'
% 
% Sangdoo Yun, 2017.

if nargin < 2
    vid_name = 'PETS';
end
if nargin < 1
    model_name = 'net_13_14_15';
end

if ~isnumeric(idnum)
    idnum=str2num(idnum);
end
if ~isnumeric(interval)
    interval=str2num(interval);
end

vid_path=fullfile('data',vid_name);

addpath('test/');
addpath(genpath('utils/'));

init_settings;

run(matconvnet_path);

load(fullfile('models', strcat(model_name,'.mat')));

opts.visualize = true;
opts.printscreen = true;

% if ~exist(fullfile(vid_path, sprintf('gt_id%01d_n%01d.txt', idnum, interval)), 'file')
[begin_frame, end_frame] = groundtruth(vid_path, idnum, interval);
% end

% frame_rate=interval*mp4toJpg(vid_pathm, 'img', vid_name);
frame_rate=15/interval;

folder_name=make_img_folder(vid_path, idnum, begin_frame, end_frame, interval);
if ~exist(fullfile(vid_path, sprintf('%s_id%01d_n%01d', model_name, idnum, interval)), 'dir')
    mkdir(fullfile(vid_path, sprintf('%s_id%01d_n%01d', model_name, idnum, interval)));
end

rng(1004);
[results, t, p] = adnet_test(net, vid_path, opts, folder_name, model_name, idnum, interval);
fprintf('precision: %f, fps: %f\n', p(20), size(results, 1)/t);

new_folder_name=sprintf('%s_id%01d_n%01d', model_name, idnum, interval);
jpgToAvi(vid_path, new_folder_name, sprintf('%s_%s', vid_name, new_folder_name), frame_rate);

