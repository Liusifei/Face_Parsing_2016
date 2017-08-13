function [img_trans,retform] = AlignHelen(img, lmk, mean_shape, ratio)
if ~exist('ratio','var') || isempty(ratio)
	ratio = 1;
end
mean_shape = ratio * mean_shape;
sizef = ratio * 250;
ptx = lmk(1:2:10);
pty = lmk(2:2:10);
[lmk_new,tform] = T1_GeneratingTrom([ptx,pty], mean_shape);
lmk_new = lmk_new + sizef/2;
[~,retform] = T1_GeneratingTrom(lmk_new',[ptx,pty]);
if size(img,3) ~= 11
	img_trans = imtransform(img,tform,'XData',[-sizef/2+1 sizef/2],...
    	'YData',[-sizef/2+1 sizef/2],'XYscale',1, 'Fillvalues',0);
else
	img_trans = imtransform(img(:,:,1),tform,'XData',[-sizef/2+1 sizef/2],...
    	'YData',[-sizef/2+1 sizef/2],'XYscale',1, 'Fillvalues',1);
	img_trans(:,:,2:11) = imtransform(img(:,:,2:11),tform,'XData',[-sizef/2+1 sizef/2],...
    	'YData',[-sizef/2+1 sizef/2],'XYscale',1, 'Fillvalues',0);
end
end
