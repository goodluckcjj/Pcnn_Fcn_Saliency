% Deep Map & PCNN segmentation Fusion

clear all;
close all;

ims = dir('../tmp2/*.png*');
path_dm = '../tmp2';
path_im = '../../input';
path_pcnn = '../tmp1';
path_out = '../../output';

Thres = 100/255;
GAMMA = 0.9;


for i = 1:length(ims)
    IMAGE = ims(i).name;
    DM_input = imread(fullfile(path_dm,IMAGE));
    PCNN_input = imread(fullfile(path_pcnn,IMAGE));
    image_input = imread(fullfile(path_im,strrep(IMAGE,'.png','.jpg')));
    [LEN,WID] = size(DM_input);
    num_pixel = numel(DM_input);
    
    % Fusion Method
    DM = im2double(DM_input);
    SEG = im2double(PCNN_input);
    
    MASK = (DM > 50/255);
    MASK = imfill(MASK,'holes');
    MASK_edge = edge(MASK,'canny');
    num_MASK_pixel = sum(sum(MASK));
    SEG_MASK = SEG.*MASK;
    SEG_WITHOUT_MASK = SEG.*(1-MASK);
    % phase opposite
    if sum(sum(SEG_MASK))/num_MASK_pixel < sum(sum(SEG_WITHOUT_MASK))/(num_pixel-num_MASK_pixel)
        SEG_MASK = (1 - SEG_MASK).*MASK;
    end
    % histogram of segmentation
    SEG_hist = imhist(SEG_MASK);
    SEG_hist = SEG_hist(2:end); % without '0' pixels
    valuethresh = find(SEG_hist>0); % PCNN result is discrete
    height = SEG_hist(valuethresh);
    
    % rearrange the gray level of SEG
    for jj = 1:length(valuethresh)
        if height(jj) > max(height)/5
            SEG_REVALUE = (SEG_MASK >= valuethresh(jj)/255) + SEG_MASK.*(SEG_MASK < valuethresh(jj)/255);
            break;
        end
    end

    Image_Fusion = (DM.^GAMMA).*(SEG_REVALUE.^(1-GAMMA));
    Image_Fusion = Image_Fusion/max(max(Image_Fusion));
%     Image_Fusion = (Image_Fusion > 1) + Image_Fusion.*(Image_Fusion <= 1);
    Image_Fusion = imfill(Image_Fusion,'holes');
    
    % calculate area coincide
    coincide = MASK.*( (Image_Fusion > Thres) == MASK );    
    if sum(sum(coincide))/sum(sum(MASK)) < 0.8
        % no enough coincide area
        % just adjust the fusion result
        % making it more tend to Deep Map
        SEG_REVALUE = imfill( DM | MASK_edge,'holes' );
%         SEG_REVALUE = SEG_REVALUE | MASK_edge;
%         SEG_REVALUE = imfill(SEG_REVALUE,'holes');
        Image_Fusion = DM.*(SEG_REVALUE.^0.9);
        Image_Fusion = (Image_Fusion > 1) + Image_Fusion.*(Image_Fusion<=1);
        Image_Fusion = imfill(Image_Fusion,'holes');
    end
    
    [labels, numlabels] = slicmex(image_input, 250, 20);
    SM_c = double(imresize(255*Image_Fusion,[LEN WID]));
    SM_refined = zeros(LEN,WID);
    for ii = 1:numlabels 
        mask = (labels==ii);
        spp_size = sum(sum(mask));
        spp_sum = sum(sum(mask.*SM_c));
        if spp_size ~= 0
            spp_avg = spp_sum/spp_size;
        else
            spp_avg = 0;
        end
        if spp_avg > 0.2
            SM_refined = SM_refined + spp_avg*mask;
        end
    end
    Image_Fusion_smooth = uint8(SM_refined);
    
    if ~mod(i,100)
        fprintf('processed: %d / %d\n',i,length(ims));
    end
    imwrite(Image_Fusion_smooth,fullfile(path_out,IMAGE));

end
