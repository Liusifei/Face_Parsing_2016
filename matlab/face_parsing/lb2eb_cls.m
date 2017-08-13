function eb = lb2eb_cls(lb, ratio)
if exist('ratio','var')
   lb = imresize(lb,ratio,'nearest');
end
[r,c,ch] = size(lb);
eb = zeros(r,c,ch);
for m = 1:ch
    tmp = lb(:,:,m);
    Y = tmp;
    Y(tmp > 0.5) = 1;
    Y(tmp <= 0.5) = 0;
    eb(:,:,m) = [zeros(1,c);double(diff(Y)>0)] + ... 
         flipud([zeros(1,c);double(diff(flipud(Y))>0)]) + ...
                [zeros(r,1),double(diff(Y,1,2)>0)] + ... 
         fliplr([zeros(r,1),double(diff(fliplr(Y),1,2)>0)]);
end
eb = 1-expand(eb);
end
