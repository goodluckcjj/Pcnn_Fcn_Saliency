% create by lczhou 2018

clear all;
close all;

if ~exist('slicmex', 'file')
    cd ./SLIC_mex
    mex slicmex.c
    cd ../
end

ims = dir('../../Input/*.jpg*');
path_img = '../../Input';
path_out = '../tmp';

SIGMA = 0.4;
THETA = 0.7;

if ~exist(path_out,'dir')
   mkdir(path_out);
   mkdir([path_out,'1']);
end

h=waitbar(0,'processing');  

ITER_MAX = 10;
beta = 0.2;
alpha_theta = 0.3;
v_theta = 20;
W = [ 0.5 1 0.5;
      1   0   1;
      0.5 1 0.5 ];
W_sp = [0 0 0.5 0 0;
        0 0.5 1 0.5 0;
        0.5 1 0 1 0.5;
        0 0.5 1 0.5 0;
        0 0 0.5 0 0];

for i = 1:length(ims)
    
IMAGE = ims(i).name;
Data_input = imread(fullfile(path_img,IMAGE));

DataG = double(rgb2gray(Data_input));
DataG = medfilt2(DataG);
[LEN, WID] = size(DataG);
Image_in = DataG/255;

%% adding white edges to image as initial on-fire points
% % not works well with some images 
% LEN = LENGTH+2;
% WID = WIDTH+2;
% Data_edge = 255*ones(LENGTH+2,WIDTH+2);
% Data_edge(2:LENGTH+1,2:WIDTH+1) = DataG;    %using gray image or H/S/V channel
% Image_in = Data_edge/255;

%% histogram of input image
Image_hist = imhist(Image_in);

% use valley
Image_hist_op = max(Image_hist) - Image_hist;
Image_hist_op_smooth = smooth(smooth(Image_hist_op));
[minv,minl] = findpeaks(Image_hist_op_smooth,'minpeakdistance',20);
%plot(Image_hist);
%hold on; plot(minl,Image_hist(minl),'*');
peak = max(Image_hist);
Thres_init = minl(minl>30 & minl<220 & minv<0.95*peak )/255;
if isempty(Thres_init) % smooth one time
    Image_hist_op_smooth = smooth(Image_hist_op);
    [minv,minl] = findpeaks(Image_hist_op_smooth,'minpeakdistance',20);
    peak = max(minv);
    Thres_init = minl(minl>10 & minl<245 & minv<0.95*peak )/255;
    if isempty(Thres_init)  % no smooth
        [minv,minl] = findpeaks(Image_hist_op,'minpeakdistance',20);
        peak = max(minv);
        Thres_init = minl(minl>10 & minl<245 & minv<0.95*peak )/255;
        if isempty(Thres_init)
            Thres_init = minl(minl>10 & minl<245 )/255;
        end
    end
end

% use peak - 10
% Image_hist_smooth = smooth(smooth(Image_hist));
% [maxv,maxl] = findpeaks(Image_hist_smooth,'minpeakdistance',20);
% Thres_init = ( maxl(maxl>20 & maxl<245 & maxv>0.1*peak) - 10)/255;

% SPP seg
[labels, numlabels] = slicmex(Data_input, 500, 20);

%% pulse-coupled neural network (simplized model)
% using multiple initial threshold
Dim = length(Thres_init);
Image_out = zeros(LEN,WID,Dim);
for kk = 1:length(Thres_init)
    % F,L,U,Y,Thresold initialize 
    F = Image_in;        %In simple PCNN model, remains unchange during iteration
    L = zeros(LEN,WID);     %Initially set to zero
    U = zeros(LEN,WID);     %Initially set to zero
    Threshold = zeros(LEN,WID) + Thres_init(kk);     %Initial value may be important
    Y = zeros(LEN,WID);     %Initially set to zero
    
    CEmin = 100;
    iter = 0;
    MI = zeros(1,ITER_MAX);
    % using simplized model, threshold decend; if fired, threshold ascend
    for ii = 1:ITER_MAX
        L = conv2(Y,W,'same');
        L = im2double(L>0);
        U = F.*(1+beta*L);
        Threshold = exp(-alpha_theta)*Threshold + v_theta*Y;
        Y_old = Y;
        Y = im2double(U>Threshold);
        % if Y_old == Y
        %   fprintf('iteration times:%d',ii);
        %   break;
        % end  

        % calculate jointHistogram
        p12 = jointHist(DataG,Y)/numel(Image_in);
        [r,c] = size(p12);
        p1 = sum(p12,2);
        p2 = sum(p12);
        H1 = 0;
        H2 = 0;
        for jj = 1:r
            if p1 ~= 0
                H1 = H1 - p1(jj)*log(p1(jj));
            end
        end
        for jj = 1:c
            if p2 ~= 0
                H2 = H2 - p2(jj)*log(p2(jj));
            end
        end
        H12 = sum(sum(-p12.*log(p12+(p12==0))));
        MI(ii) = H1 + H2 - H12;
    end

    [MImax, iter] = max(MI);
    
   if iter < 5 || length(Thres_init) == 1
    % using the calculated iteration times
    L = zeros(LEN,WID);     %Initially set to zero
    U = zeros(LEN,WID);     %Initially set to zero
    Threshold = zeros(LEN,WID) + Thres_init(kk);     %Initial value may be important
    Y = zeros(LEN,WID);     %Initially set to zero

    for ii = 1:iter
        L = conv2(Y,W,'same');
        L = im2double(L>0);
        U = F.*(1+beta*L);
        Threshold = exp(-alpha_theta)*Threshold + v_theta*Y;
        Y_old = Y;
        Y = im2double(U>Threshold);
    end     
    % to overcome phase opposition
    inner = sum(sum(Y(round(LEN/8):round(7*LEN/8),round(WID/8):round(7*WID/8))));
    outer = sum(sum(Y)) - inner;
    if inner/(round(LEN*3/4)*round(WID*3/4)) < outer/(LEN*WID-round(LEN*3/4)*round(WID*3/4))
        Y = ones(LEN,WID) - Y;
    end
    Image_out(:,:,kk) = Y ;
   else 
    Image_out(:,:,kk) = zeros(LEN,WID);
    Dim = Dim - 1;
   end
end
    
PCNN_final = sum(Image_out,3)/Dim;
% % to overcome phase opposition
% inner = sum(sum(Image_final(round(LEN/4):round(3*LEN/4),round(WID/4):round(3*WID/4))));
% outer = sum(sum(Image_final)) - inner;
% oppose = 'false';
% if inner/(round(LEN/2)*round(WID/2)) < outer/(LEN*WID-round(LEN/2)*round(WID/2))
%     Image_final = ones(LEN,WID) - Image_final;
%     oppose = 'true';
% end

center_weight = ones(WID, LEN);
for ii=1:LEN
    for jj=1:WID
        d = sqrt((ii/LEN-0.5)^2 + (jj/WID-0.5)^2);
        center_weight(ii,jj) = min(exp(1-d/SIGMA),1);
    end
end
PCNN_final = PCNN_final.*center_weight;

SM_c = double(PCNN_final);
PCNN_smooth = zeros(LEN,WID);
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
        PCNN_smooth = PCNN_smooth + spp_avg*mask;
    end
end

out_name = strrep(ims(i).name,'.jpg','.png');
Final_weight = THETA+(1-THETA)*PCNN_smooth;
out = zeros(LEN,WID,3);
for ii=1:3
    out(:,:,ii) = double(Data_input(:,:,ii)).*Final_weight;
end
imwrite(uint8(out), fullfile(path_out,ims(i).name));
imwrite(uint8(255*PCNN_smooth), fullfile([path_out,'1'],out_name))

info=[num2str(i/length(ims)*100),'%...'];
waitbar(i/length(ims),h,info)
end

delete(h)

