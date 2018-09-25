function ims = get_masked_image_v2(im, boxes)
% GET_EXTRACT_REGIONS extract image regions 
% modified from MDNET_EXTRACT_REGIONS() of MDNet 
% 
% Sangdoo Yun, 2017.

crop_size = 112;
im_ = imresize(im, [crop_size crop_size]);

num_boxes=size(boxes,1);
ims = zeros(crop_size,crop_size,6,num_boxes);

for b = 1:size(boxes, 1)
    big_mask = zeros(crop_size, crop_size, 3, 'uint8');
    mask = zeros(crop_size, crop_size, 3, 'uint8');
    
    for id = 1:3
        
        box=boxes(b,:,id);
        box([1 3])=single(round(box([1 3]) / size(im,2) * crop_size));
        box([2 4])=single(round(box([2 4]) / size(im,1) * crop_size));
        
        if box(1)<1
            box(3)=box(3)+box(1)-1;
        end
        if box(2)<1
            box(4)=box(4)+box(2)-1;
        end
        
        box(1)=max(1,box(1));
        box(2)=max(1,box(2));
        box(3)=min(crop_size-box(1),box(3));
        box(4)=min(crop_size-box(2),box(4));
        
        mask(box(2):box(2)+box(4), box(1):box(1)+box(3), id)=1;
        big_mask(box(2):box(2)+box(4), box(1):box(1)+box(3), :)=1;
    end
    
    masked_im=big_mask.*im_;
%     imshow(masked_im);
    masked_im=cat(3, masked_im, mask);
    
    ims(:,:,:,b)=masked_im;
end
