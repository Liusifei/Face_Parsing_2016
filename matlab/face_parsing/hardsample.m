function mask = hardsample(delta, channel, dsize, rate)
mask = zeros(dsize);
num = round(rate * (dsize(1)*dsize(2)));
for m = 1:dsize(4)
    msk = zeros(dsize(1),dsize(2));
    det = sum(abs(delta(:,:,:,m)),3);
    [~,I] = sort(det(:), 1, 'descend');
    msk(I(1:num)) = 1;
    mask(:,:,channel,m) = repmat(msk,[1,1,length(channel)]);
end