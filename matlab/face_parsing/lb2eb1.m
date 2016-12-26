function eb = lb2eb1(lb, ratio)
if exist('ratio','var')
   lb = imresize(lb,ratio,'nearest');
end
[r,c,ch] = size(lb);
%eb = zeros(r,c,2);
[~, Y] = max(lb, [], 3);
% y with 2 pixel
eb_y = [zeros(1,c);double(diff(Y)>0)] + ...
    flipud([zeros(1,c);double(diff(flipud(Y))>0)]);
% x with 2 pixel
eb_x = [zeros(r,1),double(diff(Y,1,2)>0)] + ...
    fliplr([zeros(r,1),double(diff(fliplr(Y),1,2)>0)]);
%eb = 1-expand(eb);
ebs = eb_x + eb_y;
%ebs(ebs>1)=1;
eb = 1 - expand(ebs);
end
