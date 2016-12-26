function [delta, loss] = Face02_L2_eb(active, eb, mode)
% edge loss for y eb(:,:,1,:) and x eb(:,:,2,:)
active = 1./(1+exp(-active));
[r,c,cha,bz] = size(active);
if size(eb,1)~= r
    eb = imresize(eb,[r,c], 'nearest');
end
dt = active - eb;
loss = 0.5 * sum(dt(:).^2)/bz;
if strcmp(mode, 'train')
    msk = uni_balance([r,c,cha,bz], 0.5);
    delta = single((msk.*dt)/bz);
else
    delta = 0;
end
end
