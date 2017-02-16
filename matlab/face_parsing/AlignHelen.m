function [img_trans,retform] = AlignHelen(img, lmk, mean_shape)
sizef = 250;
ptx = lmk(1:2:10);
pty = lmk(2:2:10);
[lmk_new,tform] = T1_GeneratingTrom([ptx,pty], mean_shape);
lmk_new = lmk_new + sizef/2;
[~,retform] = T1_GeneratingTrom(lmk_new,[ptx,pty]);
img_trans = imtransform(img,tform,'XData',[-sizef/2+1 sizef/2],...
    'YData',[-sizef/2+1 sizef/2],'XYscale',1, 'Fillvalues',0);
end