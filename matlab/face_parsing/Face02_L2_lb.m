function [delta, loss] = Face02_L2_lb(active, lb, mode)
% segment loss
%active = 1./(1+exp(-active));
active = T1_softmax(active);
[r,c,cha,bz] = size(active);
if size(lb,1)~= r
    lb = imresize(lb,[r,c], 'nearest');
end
dt = active - lb;
loss = 0.5 * sum(dt(:).^2)/bz;
if strcmp(mode, 'train')
    msk1 = hardsample(dt, 1:cha, [r,c,cha,bz], 0.2);
    msk2 = uni_balance([r,c,cha,bz], 0.3);
    msk = max(cat(5,msk1,msk2),[],5);
    delta = single((msk.*dt)/bz);
else
    delta = 0;
end
end
