function [ feat, ims ] = get_conv_feature_v2(net, img, boxes, opts)
% GET_CONV_FEATURE extract convolutional feature of image patches
% 
% Sangdoo Yun, 2017.

opts.input_size = 112;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;
opts.batchSize_test = 128;

ims = get_masked_image_v2(img, boxes);

ims = single(ims);

inputs = {'input', gpuArray(ims)};    
net.eval(inputs);
f = net.vars(end).value;
f = gather(f);
feat = f;
