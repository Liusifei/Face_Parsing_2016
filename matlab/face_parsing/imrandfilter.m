function imgn = imrandfilter(img)
[h,w,k] = size(img);

R=im2double(img(:,:,1));
G=im2double(img(:,:,2));
B=im2double(img(:,:,3));
M = 0.7*rand(3);
rR=R*M(1,1)+G*M(1,2)+B*M(1,3);
rG=R*M(2,1)+G*M(2,2)+B*M(2,3);
rB=R*M(3,1)+G*M(3,2)+B*M(3,3);

randR=rand()*0.5+0.5;
randG=rand()*0.5+0.5;
randB=rand()*0.5+0.5;

imgn=zeros(h,w,k);
imgn(:,:,1)=randR*R+(1-randR)*rR;
imgn(:,:,2)=randG*G+(1-randG)*rG;
imgn(:,:,3)=randB*B+(1-randB)*rB;

if rand > 0.7
    imgn(:,:,1) = imadjust(imgn(:,:,1),[0.2 0.8],[]);
    imgn(:,:,2) = imadjust(imgn(:,:,2),[0.2 0.8],[]);
    imgn(:,:,3) = imadjust(imgn(:,:,3),[0.2 0.8],[]);
end

if rand > 0.7
    imgn = imnoise(imgn,'gaussian',0.005);
end
