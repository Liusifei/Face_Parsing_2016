function vis = vishelen(im, big_gc11)

[r,c,~] = size(im);
imcolor11 = zeros(size(im));
cls11 = 11;
cls10 = 10;
%% color
color = [0 0 0; %background
    255 255 0; % skin
    139 76 57;% brow01
    139 54 38;% brow02
    0 205 0;% eye01
    0 138 0;%eye02
    154 50 205;%nose
    0 0 139; % l lip
    255 165 0; % m
    72 118 255 % u lip
    255 0 0]; % hair

for m = 1:cls11
    imcolor11 = imcolor11 + ...
        (repmat(reshape(color(m,:),[1,1,3]),[r c 1]).*ones(r,c,3))/255.*...
        repmat(big_gc11(:,:,m),[1,1,3]);
end
vis11 = 0.6*imcolor11 + 0.4*im;
vis11 = vis11 .* repmat(1-big_gc11(:,:,1),[1,1,3]);
a11 = (repmat(reshape(color(cls11,:),[1,1,3]),[r c 1]).*ones(r,c,3))/255.*...
    repmat(big_gc11(:,:,cls11),[1,1,3]);
vis = vis11 + 0.6* a11;
